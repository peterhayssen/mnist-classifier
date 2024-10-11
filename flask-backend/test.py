import torch
from model import ConvNet 

model = ConvNet() 
model = ConvNet()  # Or MLPNet if you're using that
model.load_state_dict(torch.load('flask-backend/models/cnn_model.pth', weights_only=True))
model.eval()