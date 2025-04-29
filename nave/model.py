import torch
import torch.nn as nn
import numpy as np

class NAVEModel(nn.Module):
    """
    Enhanced neural network for audio processing with better noise suppression
    and preservation of speech characteristics.
    """
    def __init__(self):
        super().__init__()
        # Encoder with skip connections
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1)
        )
        
        # Processing blocks
        self.processing = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1)
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        
        # Processing
        residual = x
        x = self.processing(x)
        x = x + residual  # Skip connection
        
        # Decoder
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convenience method for processing numpy arrays"""
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        enhanced = self(audio_tensor)
        return enhanced.squeeze().cpu().numpy()