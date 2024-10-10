import argparse
import torch
from model import MLPNet, ConvNet
from data_loader import get_data_loaders

if __name__ == "__main__":
    # Get data loaders
    trainloader, testloader = get_data_loaders(batch_size=64)

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose the model (MLPNet or ConvNet)
    model_choice = 'mlp'
    
    if model_choice == 'mlp':
        model = MLPNet()
    elif model_choice == 'cnn':
        model = ConvNet()
    else:
        raise ValueError("Invalid model choice! Use 'mlp' or 'cnn'.")

    # Move the model to the appropriate device (CPU or GPU)
    model = model.to(device)

    # Train the model
    train_model(model, trainloader, testloader, epochs=5, lr=0.001)