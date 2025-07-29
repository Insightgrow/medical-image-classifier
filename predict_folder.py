import os
import csv
import torch
from torchvision import models, transforms
from PIL import Image
from utils import preprocess_image, predict_image  # Make sure these exist

# Load the trained model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("resnet18_medical_classifier.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Batch prediction function
def predict_folder(folder_path, output_csv="results.csv"):
    model = load_model()

    transform = preprocess_image()

    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                try:
                    image = Image.open(image_path).convert("RGB")
                    image = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        output = model(image)
                        probs = torch.softmax(output, dim=1).numpy()[0]
                        label = "medical" if probs[0] > probs[1] else "non_medical"
                        results.append([file, label, probs[0], probs[1]])
                        print(f"{file} → {label} | confidence: {probs}")
                except Exception as e:
                    print(f" Failed to process {file}: {e}")

   
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence_medical", "confidence_non_medical"])
        writer.writerows(results)

    print(f"\nPredictions saved to {output_csv}")


if __name__ == "__main__":
    folder_path = "data"
    predict_folder(folder_path)
