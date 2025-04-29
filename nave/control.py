import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import librosa
from scipy.signal import butter, sosfilt
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Tuple, List, Union, Any
import warnings

from nave.utils import MelScaleFilterbank, QuantumNoiseStateMatrix
from nave.noise_estimator import QuantumNoiseEstimator
from nave.model import NAVEModel

class GainController:
    def __init__(self, fs: int = 16000, block_size: int = 1024, n_fft: int = 512, n_mels: int = 64, 
                 use_enhancement: bool = False, use_noise_reduction: bool = False):
        if not (8000 <= fs <= 48000):
            raise ValueError("Sample rate (fs) must be between 8000 and 48000 Hz")
        if not (block_size >= 256 and block_size <= 4096):
            raise ValueError("Block size must be between 256 and 4096")
        if not (n_fft >= 256):
            raise ValueError("n_fft must be >= 256")
        if not (16 <= n_mels <= 128):
            raise ValueError("n_mels must be between 16 and 128")

        self.fs = fs
        self.block_size = block_size
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.use_enhancement = use_enhancement
        self.use_noise_reduction = use_noise_reduction

        self._init_parameters()
        self._build_filters()
        self._init_state()
        self._setup_components()

        print(f"GainController initialized: fs={fs}, block_size={block_size}, n_fft={n_fft}, n_mels={n_mels}, "
              f"enhancement={use_enhancement}, noise_reduction={use_noise_reduction}")

    def _init_parameters(self) -> None:
        self.nominal_rms = 0.2
        self.max_gain = 15.0
        self.min_gain = 0.2
        self.noise_floor_threshold = 0.005
        self.attack_time = 0.02
        self.release_time = 0.1
        self.learning_frames = int(0.2 * self.fs / self.block_size)
        self.gain_smoothing_alpha = 0.9
        self.stft_hop_length = self.n_fft // 4
        self.short_alpha = 1.0 - np.exp(-1.0 / (self.fs * self.block_size * 0.1))
        self.long_alpha = 1.0 - np.exp(-1.0 / (self.fs * self.block_size * 1.0))

    def _build_filters(self) -> None:
        self.hp_sos = butter(6, 80, btype='highpass', fs=self.fs, output='sos')
        self.lp_sos = butter(6, 7500, btype='lowpass', fs=self.fs, output='sos')

    def _init_state(self) -> None:
        self.current_gain = 1.0
        self.gain_history = np.ones(20, dtype=np.float32)
        self.smoothed_gain = 1.0
        self.processed_frames_count = 0
        self.noise_profile = None
        self.short_term_rms = 0.0
        self.long_term_rms = 0.0

    def _setup_components(self) -> None:
        self.quantum_estimator = QuantumNoiseEstimator(
            fs=self.fs,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            smoothing=0.9
        )
        self.model = NAVEModel() if self.use_enhancement else None
        if self.model is not None:
            self.model.eval()
        self.attack_alpha = 1.0 - np.exp(-1.0 / (self.fs * self.attack_time)) if self.fs > 0 and self.attack_time > 0 else 0.1
        self.release_alpha = 1.0 - np.exp(-1.0 / (self.fs * self.release_time)) if self.fs > 0 and self.release_time > 0 else 0.01

    def safe_process(self, frame: np.ndarray) -> np.ndarray:
        try:
            if frame.size != self.block_size or frame.ndim != 1:
                frame = np.pad(frame, (0, self.block_size - frame.size), mode='constant')[:self.block_size]

            frame -= np.mean(frame)

            rms_in = np.sqrt(np.mean(frame**2))
            self.short_term_rms = (1 - self.short_alpha) * self.short_term_rms + self.short_alpha * rms_in
            self.long_term_rms = (1 - self.long_alpha) * self.long_term_rms + self.long_alpha * rms_in

            if rms_in < self.noise_floor_threshold:
                return np.zeros(self.block_size, dtype=np.float32)

            filtered_frame = sosfilt(self.hp_sos, frame)
            filtered_frame = sosfilt(self.lp_sos, filtered_frame)

            if self.use_noise_reduction:
                filtered_frame = self._reduce_noise(filtered_frame)

            processed_frame = filtered_frame
            if self.use_enhancement and self.model is not None:
                processed_frame = self._enhance_spectrum(filtered_frame)

            processed_frame = self._apply_dynamics(processed_frame)
            final_output = np.clip(processed_frame, -1.0, 1.0)

            final_output -= np.mean(final_output)

            if not np.all(np.isfinite(final_output)):
                return np.zeros(self.block_size, dtype=np.float32)
            print(f"Final output mean: {np.mean(final_output)}, RMS: {np.sqrt(np.mean(final_output**2))}")

            return final_output

        except Exception as e:
            print(f"Error in safe_process: {e}")
            return np.zeros(self.block_size, dtype=np.float32)

    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        if self.processed_frames_count < self.learning_frames:
            stft_result = librosa.stft(frame, n_fft=self.n_fft, hop_length=self.stft_hop_length)
            magnitude = np.abs(stft_result)
            if self.noise_profile is None:
                self.noise_profile = np.mean(magnitude, axis=1, keepdims=True)
            else:
                self.noise_profile = 0.95 * self.noise_profile + 0.05 * np.mean(magnitude, axis=1, keepdims=True)
            self.processed_frames_count += 1
            return frame

        stft_result = librosa.stft(frame, n_fft=self.n_fft, hop_length=self.stft_hop_length)
        magnitude = np.abs(stft_result)
        phase = np.angle(stft_result)

        if self.noise_profile is None or self.noise_profile.shape[0] != (self.n_fft // 2 + 1):
            linear_noise_estimate = magnitude * 0.1
        else:
            linear_noise_estimate = np.maximum(gaussian_filter1d(self.noise_profile[:, 0], sigma=1)[:, np.newaxis], 1e-8)
            if linear_noise_estimate.shape[1] == 1 and magnitude.shape[1] > 1:
                linear_noise_estimate = np.tile(linear_noise_estimate, (1, magnitude.shape[1]))

        subtracted_magnitude = magnitude - linear_noise_estimate
        cleaned_magnitude = np.maximum(subtracted_magnitude, magnitude * 0.05)
        clean_stft = cleaned_magnitude * np.exp(1j * phase)
        return librosa.istft(clean_stft, hop_length=self.stft_hop_length, length=len(frame)).astype(np.float32)

    def _enhance_spectrum(self, frame: np.ndarray) -> np.ndarray:
        if not self.use_enhancement or self.model is None:
            return frame

        frame_contiguous = np.ascontiguousarray(frame, dtype=np.float32)
        tensor = torch.from_numpy(frame_contiguous).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            enhanced_tensor = self.model(tensor)

        enhanced_frame = enhanced_tensor.squeeze().cpu().numpy()
        if enhanced_frame.shape[0] != frame.shape[0]:
            out_len = enhanced_frame.shape[0]
            target_len = frame.shape[0]
            if out_len < target_len:
                enhanced_frame = np.pad(enhanced_frame, (0, target_len - out_len))
            else:
                enhanced_frame = enhanced_frame[:target_len]

        return enhanced_frame.astype(np.float32)

    def _apply_dynamics(self, frame: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(frame**2))
        if rms < 1e-8:
            return frame

        target_rms = self.nominal_rms * max(1.0, self.long_term_rms / self.nominal_rms)
        target_gain = np.clip(target_rms / rms if rms > 0 else 1.0, self.min_gain, self.max_gain)

        if target_gain < 1.0 and rms > 0.5 * self.nominal_rms:
            target_gain = 1.0 - 0.5 * (1.0 - target_gain) * (rms / (0.5 * self.nominal_rms) - 1.0)

        alpha = self.attack_alpha if target_gain < self.current_gain else self.release_alpha
        self.gain_history = np.roll(self.gain_history, -1)
        self.gain_history[-1] = target_gain
        self.current_gain = (1 - alpha) * self.current_gain + alpha * np.mean(self.gain_history)

        self.smoothed_gain = self.gain_smoothing_alpha * self.smoothed_gain + (1 - self.gain_smoothing_alpha) * self.current_gain
        return frame * self.smoothed_gain

    def start_stream(self, callback: callable) -> None:
        if hasattr(self, 'stream') and self.stream.active:
            print("Stream already running.")
            return
        try:
            self.stream = sd.InputStream(
                samplerate=self.fs,
                blocksize=self.block_size,
                device=sd.default.device[0],
                channels=1,
                dtype='float32',
                callback=lambda indata, frames, time, status: callback(indata[:, 0], frames, time, status),
                latency='low'
            )
            self.stream.start()
        except Exception as e:
            print(f"Error starting audio stream: {str(e)}")
            if hasattr(self, 'stream'):
                self.stream.close()
                self.stream = None
            raise

    def stop_stream(self) -> None:
        if hasattr(self, 'stream') and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        else:
            print("No active stream found to stop.")

    def reset(self) -> None:
        self.stop_stream()
        self._init_state()