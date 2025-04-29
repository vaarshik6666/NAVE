import numpy as np
from typing import Optional

# Version of MelScaleFilterbank originally from utils.py
# Choose this OR the one from control.py based on your needs
class MelScaleFilterbank:
    def __init__(self, n_fft: int = 512, fs: int = 16000, n_mels: int = 64):
        self.n_fft = max(256, n_fft)
        self.fs = max(8000, fs)
        self.n_mels = max(16, min(n_mels, 128)) # Ensure n_mels is reasonable
        self.filters = self._build_filters() # Store filters directly

    def _hz_to_mel(self, freq: float) -> float:
        """Converts Hz to Mel scale."""
        return 2595 * np.log10(1 + freq / 700)

    def _mel_to_hz(self, mel: float) -> float:
        """Converts Mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    def _build_filters(self) -> np.ndarray:
        """Builds the triangular Mel filterbank."""
        # Define frequency range (e.g., 20Hz to Nyquist)
        low_freq_mel = self._hz_to_mel(20)
        high_freq_mel = self._hz_to_mel(self.fs / 2)

        # Create evenly spaced points in Mel scale
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        # Convert Mel points back to Hz
        hz_points = self._mel_to_hz(mel_points)

        # Convert Hz points to FFT bin indices
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.fs).astype(int)
        # Ensure bins are within valid range [0, n_fft/2]
        bin_points = np.clip(bin_points, 0, self.n_fft // 2)

        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1), dtype=np.float32)

        for i in range(self.n_mels):
            # Indices for the start, center, and end of the triangular filter
            start_bin = bin_points[i]
            center_bin = bin_points[i + 1]
            end_bin = bin_points[i + 2]

            # Build the left slope (ascending)
            if center_bin > start_bin:
                filters[i, start_bin:center_bin] = np.linspace(0, 1, center_bin - start_bin, endpoint=False, dtype=np.float32)
            # Build the right slope (descending)
            if end_bin > center_bin:
                filters[i, center_bin:end_bin] = np.linspace(1, 0, end_bin - center_bin, endpoint=False, dtype=np.float32)

        # Normalize filters (optional, common practice)
        # Each filter sums to 1, or normalize by bandwidth if preferred
        # Normalization disabled here, similar to control.py version
        # filter_sums = filters.sum(axis=1, keepdims=True)
        # filters = np.where(filter_sums > 1e-8, filters / filter_sums, filters)

        return filters

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Applies the Mel filterbank to an audio frame's spectrum."""
        # Ensure frame is float32 and has enough length for FFT
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        if len(frame) < self.n_fft:
            # Pad with zeros if shorter than n_fft
            frame = np.pad(frame, (0, self.n_fft - len(frame)))
        elif len(frame) > self.n_fft:
             # Truncate if longer (or use windowing before FFT)
             frame = frame[:self.n_fft]


        # Compute the magnitude spectrum (using rfft for real input)
        spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))

        # Ensure spectrum length matches filter dimensions
        if spectrum.shape[0] != self.filters.shape[1]:
             # This indicates a mismatch in n_fft calculation or rfft output length
             print(f"Warning: Spectrum length ({spectrum.shape[0]}) != Filter length ({self.filters.shape[1]})")
             # Adjust spectrum or filters if possible, or raise error
             # Simple truncation/padding (use with caution):
             spec_len = spectrum.shape[0]
             filt_len = self.filters.shape[1]
             if spec_len < filt_len:
                 spectrum = np.pad(spectrum, (0, filt_len - spec_len))
             else:
                 spectrum = spectrum[:filt_len]


        # Apply the filterbank by matrix multiplication (dot product)
        # Result is Mel spectrogram energies for this frame
        mel_energies = np.dot(self.filters, spectrum)

        # Apply floor to avoid log(0) or small number issues later
        mel_energies = np.maximum(mel_energies, 1e-8)

        return mel_energies


# Version of QuantumNoiseStateMatrix copied from control.py
# This now has the necessary methods.
class QuantumNoiseStateMatrix:
    def __init__(self, n_mels: int = 64, smoothing: float = 0.85): # Added n_mels argument
        self.n_mels = n_mels # Store n_mels
        self.state: Optional[np.ndarray] = None
        self.smoothing = np.clip(smoothing, 0.5, 0.99)
        # Use n_mels for initialization based on filterbank output size
        self.quantum_phases = np.random.uniform(0, 2*np.pi, self.n_mels).astype(np.float32)
        self.decoherence_factor = 0.95
        self.quantum_entanglement = np.random.uniform(0, 0.1, self.n_mels).astype(np.float32) # Initialize properly
        self.min_state_value = 1e-8

    def apply_rotation(self, energy_dist: np.ndarray) -> None:
        energy_dist = np.ascontiguousarray(energy_dist, dtype=np.float32)
        # Ensure input matches expected size
        if energy_dist.shape[0] != self.n_mels:
             print(f"Warning: energy_dist shape {energy_dist.shape} != n_mels {self.n_mels}. Resizing/Padding.")
             # Handle mismatch: Pad or truncate (choose based on desired behavior)
             if energy_dist.shape[0] < self.n_mels:
                 energy_dist = np.pad(energy_dist, (0, self.n_mels - energy_dist.shape[0]), mode='edge') # Pad with edge values
             else:
                 energy_dist = energy_dist[:self.n_mels] # Truncate


        energy_dist = np.clip(energy_dist, self.min_state_value, None)

        if self.state is None:
            # Initialize state on first call
            self.state = energy_dist.copy()
            # Entanglement is already initialized in __init__
        else:
             # Ensure state also matches n_mels if it was somehow initialized differently
             if self.state.shape[0] != self.n_mels:
                  print(f"Warning: self.state shape {self.state.shape} != n_mels {self.n_mels}. Reinitializing state.")
                  self.state = energy_dist.copy() # Re-initialize if mismatch

             # Now all arrays (state, energy_dist, phases, entanglement) should have length n_mels
             phase_shift = np.sin(self.quantum_phases + self.quantum_entanglement)

             adjusted_energy = energy_dist * (1 + 0.15 * phase_shift)
             self.state = (self.smoothing * self.state +
                           (1 - self.smoothing) * adjusted_energy)

             # Update entanglement state
             self.quantum_entanglement = (
                 self.decoherence_factor * self.quantum_entanglement +
                 0.05 * np.random.randn(self.n_mels) # Use n_mels
             )
             # Optional: Bound entanglement to prevent runaway values
             self.quantum_entanglement = np.clip(self.quantum_entanglement, -np.pi, np.pi)


    def get_noise_floor(self) -> np.ndarray:
        if self.state is None:
            # Return a default floor if state not yet initialized
            return np.full(self.n_mels, self.min_state_value, dtype=np.float32)

        # Check consistency before use
        if self.state.shape[0] != self.n_mels or self.quantum_phases.shape[0] != self.n_mels:
             print(f"Warning: Mismatch in get_noise_floor shapes. State: {self.state.shape}, Phases: {self.quantum_phases.shape}, n_mels: {self.n_mels}")
             # Handle error: return default or try to use min length? Returning default is safer.
             return np.full(self.n_mels, self.min_state_value, dtype=np.float32)

        # Calculate noise floor based on current state and phases
        noise_floor = self.state * (1 + 0.05 * np.sin(self.quantum_phases))

        # Clip to minimum value
        return np.clip(noise_floor, self.min_state_value, None)