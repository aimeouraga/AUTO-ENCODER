# Imports
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from DeepLinearAutoencoder import DeepLinearAutoencoder
from SimpleLinearAutoEncoder import LinearAutoencoder

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import train_loader and test_loader from the data module
from utils.data import train_loader, test_loader


def train(num_epochs, criterion, optimizer, model):

  for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
def reconstruct():
    
  with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        output = model(img)

        # Plotting the first 8 test images and their reconstructions
        fig, axes = plt.subplots( 2, 8, figsize=(12, 3))
        for i in range(8):
            axes[0, i].imshow(img[i].view(28, 28).numpy(), cmap='gray')
            axes[1, i].imshow(output[i].view(28, 28).numpy(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        plt.show()
        break


if __name__ == "__main__":
    
    criterion = nn.MSELoss()
    num_epochs = 10
    
    model = DeepLinearAutoencoder()
    print('='*40,f'DEEP LINEAR AUTO-ENCODER','='*40)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train(num_epochs, criterion, optimizer, model)
    reconstruct()
    
    
    print('='*40,f'LINEAR AUTO-ENCODER','='*40)
    model = LinearAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train(num_epochs, criterion, optimizer, model)
    reconstruct()
    
    

