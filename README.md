# Medical Image Classifier

This is a foundational AI/ML project that classifies images as **Medical** or **Non-Medical**. It supports both direct folder inputs and image extraction from PDFs, making it useful for real-world scenarios like scanning reports or validating image data.

---

##  What It Does

-  Uses a pre-trained ResNet-18 model to classify images.
-  Accepts a folder of images or a PDF (extracts images from it).
-  Saves results with predictions and confidence scores in `results.csv`.

---

## Project Structure
``` text
medical_image_classifier/
│
├── extract_and_classify.py # Extracts images from PDF, classifies and logs results
├── main.py # Predicts on a single image or folder
├── model.py # Contains model architecture and loader
├── predict_folder.py # Classifies all images in a given folder
├── utils.py # Helper functions (e.g., image transforms)
├── resnet18_medical_classifier.pth # Pretrained model weights
├── results.csv # Output file with predictions
├── requirements.txt # Python dependencies
├── report.md # Project overview (see separately)
├── sample_medical.pdf # Sample input PDF for demo
├── .gitignore # Files and folders to ignore
└── venv/ # Python virtual environment (excluded from Git)

Requirements
- Python 3.8+
- PyTorch
- torchvision
- Pillow
- PyMuPDF (fitz)
- tqdm
- OpenCV
