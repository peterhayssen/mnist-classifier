from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from model import ConvNet 

app = Flask(__name__)
CORS(app, origins=["https://peterhayssen.github.io"])

# Load the trained model
model = ConvNet()  # Or MLPNet if you're using that
model.load_state_dict(torch.load('flask-backend/models/cnn_model.pth', weights_only=True))
model.eval()

def preprocess_image(img_data):
    img_data = re.sub('^data:image/.+;base64,', '', img_data)
    img = Image.open(BytesIO(base64.b64decode(img_data))).convert('L')
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize the pixel values
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape to (1, 1, 28, 28)
    return img

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    img_data = data['image']
    img = preprocess_image(img_data)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        result = predicted.item()

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)