# Imports
import torch
from torchvision import datasets, transforms

# Loading data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=64, shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=64, 
    shuffle=False,
)