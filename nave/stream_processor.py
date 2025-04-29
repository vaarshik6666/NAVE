import numpy as np

class AudioStreamProcessor:
    def __init__(self, block_size=1024, overlap=0.5):
        self.block_size = block_size
        self.overlap = overlap
        self.hop_size = int(block_size * (1 - overlap))
        self.buffer = np.zeros(block_size + self.hop_size * 3)
        self.pointer = self.block_size  # Start at block_size to avoid negative indices
        
    def process(self, frame):
        if len(frame) != self.block_size:
            frame = np.pad(frame, (0, self.block_size - len(frame)))
            
        if self.pointer + self.block_size > len(self.buffer):
            self.buffer = np.roll(self.buffer, -self.hop_size)
            self.pointer -= self.hop_size
            self.pointer = max(self.block_size, self.pointer)
            
        self.buffer[self.pointer:self.pointer+self.block_size] = frame
        self.pointer += self.hop_size
        start_idx = max(0, self.pointer - self.block_size)
        return self.buffer[start_idx:self.pointer]