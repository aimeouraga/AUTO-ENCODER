import torch.nn as nn
import torch

# Define the Contractive Autoencoder class
class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, lambda_):
        super(ContractiveAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Assuming input data is normalized between 0 and 1
        )

        self.lambda_ = lambda_
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    
    def contractive_loss(self, inputs, reconstruction, latent):
        criterion = nn.MSELoss()
        mse_loss = criterion(reconstruction, inputs)
        
        # Compute the Jacobian matrix J with respect to the latent representation
        latent_grad = torch.autograd.grad(outputs=latent, inputs=inputs,
                                          grad_outputs=torch.ones_like(latent),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        # Compute the Frobenius norm of the Jacobian
        J_frobenius = torch.norm(latent_grad.view(latent_grad.size(0), -1), dim=1)
        contractive_loss = torch.sum(J_frobenius ** 2)
        
        # Total loss
        total_loss = mse_loss + self.lambda_ * contractive_loss
        return total_loss