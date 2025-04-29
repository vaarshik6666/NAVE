# nave/noise_estimator.py

import numpy as np
from typing import Tuple

# Import the classes from the utils module where they should be defined
from .utils import MelScaleFilterbank, QuantumNoiseStateMatrix

class QuantumNoiseEstimator:
    """
    Estimates the noise floor based on Mel-scale energy distribution
    using a custom 'quantum state matrix' approach for smoothing and variation.
    """
    def __init__(self,
                 fs: int = 16000,
                 n_fft: int = 512,
                 n_mels: int = 64,
                 smoothing: float = 0.9):
        """
        Initializes the QuantumNoiseEstimator.

        Args:
            fs (int): Sample rate of the audio.
            n_fft (int): FFT size used for analysis. This determines the frame size.
            n_mels (int): Number of Mel filterbanks to use.
            smoothing (float): Smoothing factor for the quantum state update (0.5 to 0.99).
        """
        if not (8000 <= fs <= 48000):
             raise ValueError("Sample rate (fs) must be between 8000 and 48000 Hz")
        if not (n_fft >= 256 and (n_fft & (n_fft - 1) == 0)): # Check power of 2 >= 256
             # Allow non-power-of-2, but warn? For now, stick to common FFT sizes.
             print(f"Warning: n_fft={n_fft} is not a power of 2 >= 256. Using {max(256, n_fft)}.")
             n_fft = max(256, n_fft) # Adjust if needed, or raise error
        if not (16 <= n_mels <= 128):
             raise ValueError("Number of Mel bands (n_mels) must be between 16 and 128")
        if not (0.5 <= smoothing <= 0.99):
             raise ValueError("Smoothing factor must be between 0.5 and 0.99")


        self.frame_size: int = n_fft # Use n_fft as the required frame size
        self.n_mels: int = n_mels
        self.fs: int = fs
        self.smoothing: float = smoothing

        # Instantiate components using the imported classes from utils
        self.filterbank = MelScaleFilterbank(n_fft=self.frame_size, fs=self.fs, n_mels=self.n_mels)
        self.quantum_state = QuantumNoiseStateMatrix(n_mels=self.n_mels, smoothing=self.smoothing)

        # Precompute the analysis window
        self.window = np.hanning(self.frame_size).astype(np.float32)

        # Minimum value floor to prevent issues with zeros
        self.min_value_floor = 1e-8

        print(f"QuantumNoiseEstimator initialized: fs={fs}, n_fft={n_fft}, n_mels={n_mels}, smoothing={smoothing}")


    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Updates the noise estimate based on the input audio frame.

        Args:
            frame (np.ndarray): Input audio frame (1D array).

        Returns:
            np.ndarray: Estimated noise floor across Mel bands (shape: (n_mels,)).
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 1:
             raise TypeError("Input 'frame' must be a 1D NumPy array.")

        # Ensure the frame is the correct data type and contiguous
        frame = np.ascontiguousarray(frame, dtype=np.float32)

        # --- Frame Length Handling ---
        current_len = frame.shape[0]
        if current_len != self.frame_size:
            # print(f"QNE: Input frame length {current_len} != required {self.frame_size}. Adjusting.") # Optional Debug
            if current_len < self.frame_size:
                # Pad with zeros if too short
                frame = np.pad(frame, (0, self.frame_size - current_len), mode='constant')
            else:
                # Truncate if too long (alternative: use overlapping windows)
                frame = frame[:self.frame_size]
        # --- End Frame Length Handling ---

        # Apply the analysis window
        windowed_frame = frame * self.window

        # Calculate Mel-scale energy distribution using the filterbank
        # Filterbank __call__ computes FFT and applies filters
        mel_energy_distribution = self.filterbank(windowed_frame)

        # Apply clipping (redundant if MelScaleFilterbank already does it, but safe)
        mel_energy_distribution = np.maximum(mel_energy_distribution, self.min_value_floor)

        # Update the internal quantum noise state using the energy distribution
        self.quantum_state.apply_rotation(mel_energy_distribution)

        # Retrieve the estimated noise floor from the quantum state
        noise_floor = self.quantum_state.get_noise_floor()

        # Ensure the noise floor is also floored (redundant if get_noise_floor does it)
        noise_floor = np.maximum(noise_floor, self.min_value_floor)

        # Expected shape is (n_mels,)
        # print(f"QNE Output noise_floor shape: {noise_floor.shape}") # Optional Debug

        return noise_floor

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    print("Testing QuantumNoiseEstimator...")

    # --- Mock Utils Classes for standalone testing ---
    # Normally these would be imported from .utils
    class MockMelScaleFilterbank:
        def __init__(self, n_fft, fs, n_mels):
            self.n_mels = n_mels
            print(f"MockMelScaleFilterbank created: n_fft={n_fft}, fs={fs}, n_mels={n_mels}")
        def __call__(self, frame):
            # Return dummy data of the correct shape
            print(f"MockMelScaleFilterbank called with frame shape: {frame.shape}")
            return np.random.rand(self.n_mels).astype(np.float32) * 0.1 + 1e-8

    class MockQuantumNoiseStateMatrix:
        def __init__(self, n_mels, smoothing):
            self.n_mels = n_mels
            self._noise_level = np.random.rand(n_mels).astype(np.float32) * 0.01 + 1e-8
            print(f"MockQuantumNoiseStateMatrix created: n_mels={n_mels}, smoothing={smoothing}")
        def apply_rotation(self, energy_dist):
             print(f"MockQuantumNoiseStateMatrix apply_rotation called with energy_dist shape: {energy_dist.shape}")
             # Simulate some state change
             self._noise_level = self._noise_level * 0.95 + energy_dist * 0.05
        def get_noise_floor(self):
            print("MockQuantumNoiseStateMatrix get_noise_floor called")
            return self._noise_level.copy()

    # Replace actual imports with mocks for the test
    import sys
    sys.modules['nave.utils'] = type('module', (object,), {
        'MelScaleFilterbank': MockMelScaleFilterbank,
        'QuantumNoiseStateMatrix': MockQuantumNoiseStateMatrix
    })()
    # --- End Mocking ---

    # Now create the estimator (it will use the mocks)
    test_fs = 16000
    test_n_fft = 512
    test_n_mels = 64
    estimator = QuantumNoiseEstimator(fs=test_fs, n_fft=test_n_fft, n_mels=test_n_mels, smoothing=0.9)

    # Create a dummy audio frame
    dummy_frame = np.random.randn(test_n_fft).astype(np.float32) * 0.1

    # Call the update method
    estimated_noise = estimator.update(dummy_frame)

    print(f"\nEstimated noise floor shape: {estimated_noise.shape}")
    print(f"Estimated noise floor (first 5 elements):\n{estimated_noise[:5]}")

    assert estimated_noise.shape == (test_n_mels,), "Output shape mismatch!"
    assert estimated_noise.dtype == np.float32, "Output dtype mismatch!"
    assert np.all(estimated_noise >= 1e-8), "Output values below floor!"

    # Test frame padding
    short_frame = np.random.randn(test_n_fft // 2).astype(np.float32) * 0.1
    estimated_noise_short = estimator.update(short_frame)
    print(f"\nEstimated noise floor shape (short frame): {estimated_noise_short.shape}")
    assert estimated_noise_short.shape == (test_n_mels,)

    # Test frame truncation
    long_frame = np.random.randn(test_n_fft * 2).astype(np.float32) * 0.1
    estimated_noise_long = estimator.update(long_frame)
    print(f"\nEstimated noise floor shape (long frame): {estimated_noise_long.shape}")
    assert estimated_noise_long.shape == (test_n_mels,)


    print("\nQuantumNoiseEstimator test completed successfully.")