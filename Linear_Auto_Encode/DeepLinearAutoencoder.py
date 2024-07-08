# Imports
import torch
import torch.nn as nn

class DeepLinearAutoencoder(nn.Module):
    """_summary_: In this implementation we will use six hidden layers.
    Latent code size to 3
    """
    def __init__(self):
        super(DeepLinearAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16), 
            nn.Linear(16, 3) 
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Linear(16, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()  # Use sigmoid activation to output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x