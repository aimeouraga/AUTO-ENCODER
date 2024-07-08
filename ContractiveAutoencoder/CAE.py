# Imports
import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import train_loader and test_loader from the data module
from utils.data import train_loader, test_loader
from ContractiveAutoencoder import ContractiveAutoencoder

# Training function    
def train_model(model, optimizer, train_loader, test_loader, num_epochs=10, threshold=0.05):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, _ in train_loader:
            inputs = inputs.view(inputs.size(0), -1)#.to(device)  # Flatten the input
            inputs.requires_grad = True  # Enable gradient computation for inputs
            
            optimizer.zero_grad()
            outputs, latent = model(inputs)
            loss = model.contractive_loss(inputs, outputs, latent)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    
def reconstruct(lambda_): 
       
    with torch.no_grad():
        std = 1.5
        for data in test_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img_noisy = img + torch.randn(img.size()) * std
            output, _ = model(img_noisy)
            # print(output.shape)
            # Plotting the first 8 test images and their reconstructions
            fig, axes = plt.subplots( 3, 8, figsize=(12, 5))
            for i in range(8):
                axes[0, i].imshow(img[i].view(28, 28).numpy(), cmap='gray')
                axes[1, i].imshow(img_noisy[i].view(28, 28).numpy(), cmap='gray')
                axes[2, i].imshow(output[i].view(28, 28).numpy(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                axes[2, i].axis('off')
            plt.show()
            break
        print('='*40,f'END RESULTS FOR lambda={lambda_}','='*40)
        print()
    
    
if __name__ == "__main__":
    # Parameters
    input_dim = 28 * 28 
    latent_dim = 64
    num_epochs = 10
    lambda_ = 5
    
    print('='*40,f'BEGINING RESULTS FOR lambda={lambda_}','='*40)
    model = ContractiveAutoencoder(input_dim, latent_dim, lambda_)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    train_model(model, optimizer, train_loader, test_loader, num_epochs)
    reconstruct(lambda_)
    
    lambda_ = 1e-4
    print('='*40,f'BEGINING RESULTS FOR lambda={lambda_}','='*40)
    model = ContractiveAutoencoder(input_dim, latent_dim, lambda_)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_model(model, optimizer, train_loader, test_loader, num_epochs)
    reconstruct(lambda_)
    
    

