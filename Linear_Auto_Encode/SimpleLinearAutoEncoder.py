# Imports
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """_summary_: In this implementation we will use five hidden layers, Latent code size to 8
    """
    def __init__(self):
        super(LinearAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 16), 
            nn.Linear(16, 8)  # Larger latent code
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Use sigmoid activation for output values ​​between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = LinearAutoencoder()