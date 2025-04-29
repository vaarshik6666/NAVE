from nave.control import GainController
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import time

class AudioGUI:
    def __init__(self, fs=16000, block_size=1024, use_enhancement=True, use_noise_reduction=True):
        self.fs = fs
        self.block_size = block_size
        self.controller = GainController(fs=fs, block_size=block_size, 
                                       use_enhancement=use_enhancement, 
                                       use_noise_reduction=use_noise_reduction)

        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('black')
            ax.grid(True, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlim(0, block_size)
            ax.set_ylim(-1, 1)

        self.raw_line = Line2D([], [], color='cyan', linewidth=1, alpha=0.8)
        self.enhanced_line = Line2D([], [], color='magenta', linewidth=1, alpha=0.8)
        self.ax1.add_line(self.raw_line)
        self.ax2.add_line(self.enhanced_line)
        
        self.ax1.set_title("Raw Input", color='white')
        self.ax2.set_title("Enhanced Output", color='white')

        self.raw_data = np.zeros(block_size)
        self.enhanced_data = np.zeros(block_size)

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=30, blit=True)
        plt.tight_layout()

    def update_plot(self, frame):
        self.raw_line.set_data(np.arange(self.block_size), self.raw_data)
        self.enhanced_line.set_data(np.arange(self.block_size), self.enhanced_data)
        return self.raw_line, self.enhanced_line

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.raw_data = indata[:, 0].copy()
        self.enhanced_data = self.controller.safe_process(self.raw_data)

    def start(self):
        print("Starting audio stream and GUI...")
        print("Close the plot window to stop the application")
        self.controller.start_stream(self.callback)
        plt.show()

    def stop(self):
        self.controller.stop_stream()
        plt.close()

if __name__ == "__main__":
    gui = AudioGUI(use_enhancement=True, use_noise_reduction=True)
    try:
        gui.start()
    except KeyboardInterrupt:
        gui.stop()