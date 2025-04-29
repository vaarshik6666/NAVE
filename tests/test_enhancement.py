import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from nave.noise_estimator import QuantumNoiseEstimator
from nave.stream_processor import AudioStreamProcessor

def test_processing():
    fs = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration))
    data = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(int(fs * duration))

    estimator = QuantumNoiseEstimator(fs=fs, n_fft=512, n_mels=64)
    processor = AudioStreamProcessor(block_size=512)

    processed = []
    for i in range(0, len(data), 512):
        frame = data[i:i+512]
        if len(frame) < 512:
            frame = np.pad(frame, (0, 512 - len(frame)), mode='constant')
        noise_level = estimator.update(frame)
        enhanced = processor.process(frame)
        processed.extend(enhanced)

    # Convert processed to numpy array and trim/pad to match original length
    processed = np.array(processed)
    if len(processed) > len(data):
        processed = processed[:len(data)]
    elif len(processed) < len(data):
        processed = np.pad(processed, (0, len(data) - len(processed)), mode='constant')

    # Plot a zoomed-in portion (first 0.1 seconds)
    zoom_samples = int(fs * 0.1)  # 0.1 seconds
    time_axis = t[:zoom_samples]

    plt.figure(figsize=(12, 8))
    
    # Original signal
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, data[:zoom_samples], color='blue')
    plt.title('Original Signal (440 Hz Sine + Noise)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.ylim(-1.0, 1.0)  # Adjust based on expected amplitude

    # Processed signal
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, processed[:zoom_samples], color='green')
    plt.title('Processed Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.ylim(-1.0, 1.0)  # Adjust based on expected amplitude

    plt.tight_layout()

    # Save the plot
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "waveform_comparison.png"))
    plt.close()

    print("Test completed successfully. Plot saved to 'plots/waveform_comparison.png'")

if __name__ == "__main__":
    test_processing()