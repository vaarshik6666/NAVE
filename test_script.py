import torch
import numpy as np
from nave import QuantumNoiseEstimator, AudioStreamProcessor, NAVEModel, GainController

# Initialize components
fs = 16000
block_size = 1024
noise_estimator = QuantumNoiseEstimator(fs=fs)
stream_processor = AudioStreamProcessor(block_size=block_size)
model = NAVEModel()
controller = GainController()

# Generate dummy audio
frame = np.random.randn(block_size)

# Process
processed_frame = stream_processor.process(frame)
noise_level = noise_estimator.update(processed_frame)
error = noise_level - 0.1
gain = controller.adjust(error)
print(f"Noise level: {noise_level}, Gain: {gain}")

# Model processing (CHANGED: Proper input shape)
frame_tensor = torch.tensor(processed_frame).float().reshape(1, 1, -1)  # [B, C, T]
enhanced_frame = model(frame_tensor)
print(f"Enhanced frame shape: {enhanced_frame.shape}")