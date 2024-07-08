import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from model import DenoisingAutoencoder
from utils import add_gaussian_noise, add_salt_and_pepper_noise, add_random_dropout_noise
from datasets import *


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = DenoisingAutoencoder().to(device)

learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
# Training loop
# losses = []
for epoch in trange(num_epochs):
    model.train()
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        noisy_img = add_random_dropout_noise(img).to(device)

        # Forward pass
        outputs = model(noisy_img)
        loss = criterion(outputs, img)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # losses.append(loss.cpu().numpy())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Testing loop
model.eval()
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.to(device)
        noisy_img = add_random_dropout_noise(img).to(device)
        outputs = model(noisy_img)



        # Visualize the results
        img = img.cpu().numpy()
        noisy_img = noisy_img.cpu().numpy()
        outputs = outputs.cpu().numpy()

        fig, axes = plt.subplots(3, 6, figsize=(12, 6))
        for i in range(6):
            axes[0, i].imshow(img[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            axes[1, i].imshow(noisy_img[i].reshape(28, 28), cmap='gray')
            axes[1, i].set_title('Noisy')
            axes[1, i].axis('off')
            axes[2, i].imshow(outputs[i].reshape(28, 28), cmap='gray')
            axes[2, i].set_title('Denoised')
            axes[2, i].axis('off')
        plt.show()
        break