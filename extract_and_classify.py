import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import fitz  
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import csv
from utils import predict_image, load_model

# Set up image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create output dir
os.makedirs("extracted_images", exist_ok=True)

def extract_images_from_pdf(pdf_path):
    print(f"Extracting images from PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    images = []
    for i in range(len(doc)):
        page = doc[i]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            img_name = f"pdf_page_{i}_img_{img_index}.{image_ext}"
            img_path = os.path.join("extracted_images", img_name)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            images.append(img_path)
    return images

def extract_images_from_url(url):
    print(f"Extracting images from URL: {url}")
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    imgs = soup.find_all('img')
    image_paths = []

    for idx, img in enumerate(imgs):
        src = img.get('src')
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = url.rstrip("/") + src
        elif not src.startswith("http"):
            continue
        try:
            img_data = requests.get(src, timeout=5).content
            img_name = f"url_img_{idx}.jpg"
            img_path = os.path.join("extracted_images", img_name)
            with open(img_path, 'wb') as handler:
                handler.write(img_data)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Failed to download {src}: {e}")
    return image_paths

def classify_images(image_paths, model):
    print("Classifying extracted images...")
    results = []
    for img_path in image_paths:
        label, confidence = predict_image(img_path, model)
        results.append((img_path, label, confidence))
    return results

def save_results_csv(results, filename="results.csv"):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Prediction", "Confidence"])
        for path, label, conf in results:
            writer.writerow([path, label, conf])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_and_classify.py <pdf_path | url>")
        sys.exit(1)

    input_path = sys.argv[1]
    model = load_model("resnet18_medical_classifier.pth")

    if input_path.lower().endswith(".pdf"):
        images = extract_images_from_pdf(input_path)
    elif input_path.startswith("http"):
        images = extract_images_from_url(input_path)
    else:
        print("Input must be a .pdf file or a valid URL")
        sys.exit(1)

    if not images:
        print("No images found.")
        sys.exit(0)

    results = classify_images(images, model)
    save_results_csv(results)
    print("Classification complete. Results saved to results.csv.")
