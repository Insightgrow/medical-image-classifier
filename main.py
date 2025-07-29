import torch
from model import build_model
from utils import predict_image

# Load model
model = build_model()
model.load_state_dict(torch.load("resnet18_medical_classifier.pth", map_location=torch.device('cpu')))

# Predict on example images
predict_image("data/medical/COVID-1017.png", model)
predict_image("data/non_medical/7145.jpg", model)
