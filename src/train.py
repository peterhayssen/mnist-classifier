import torch
import torch.optim as optim
import torch.nn as nn
from model import MLPNet, ConvNet 
from data_loader import get_data_loaders

def train_model(model, trainloader, testloader, epochs=5, lr=0.001):
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

