from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

# Preprocessing function
def preprocess_image():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# Prediction function for a single image
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    transform = preprocess_image()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).numpy()
        label = "medical" if probs[0][0] > probs[0][1] else "non_medical"
        confidence = probs[0][0] if label == "medical" else probs[0][1]
        print(f"{image_path} → {label} | confidence: {probs}")
        return label, confidence


# Load the model
def load_model(model_path="resnet18_medical_classifier.pth"):
    model = models.resnet18(pretrained=False)  # ✅ Compatible with older torchvision
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
