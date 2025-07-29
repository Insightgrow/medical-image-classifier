# Project Report

## Project Title:
**Medical VS Non-Medical Image Classifier**

---

## Abstract:
This is a deep learning-based image classifier designed to distinguish between **medical** and **non-medical** images.The system uses a transfer-learned ResNet18 model trained on a custom-labeled dataset to classify images with high accuracy. It supports inputs via local folders, PDFs, and URLs. The output is stored in a CSV format for further analysis.

---

## Tools & Technologies Used:
- **Programming Language**: Python
- **Frameworks**: PyTorch, torchvision
- **Libraries**:
  - `OpenCV` – image preprocessing
  - `pdf2image` – extract images from PDFs
  - `pytesseract` (future use) – OCR for scanned text
  - `scikit-learn` – evaluation metrics
- **Model**: ResNet18 (transfer learning from ImageNet)

---

## Model Architecture:
- **Backbone**: ResNet18 (pre-trained on ImageNet)
- **Task**: Binary classification – Medical vs Non-Medical
- **Layers Modified**: Final fully connected layer to 2 outputs
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

## How It Works:
1. **Input**:
   - A folder of image files (`.jpg`, `.png`, etc.)
   - A PDF file containing image content
   - A URL (from which a screenshot is extracted)
2. **Processing**:
   - Images are resized and normalized
   - Passed through the trained CNN model
3. **Output**:
   - Prediction printed on terminal
   - All results saved to `predictions.csv`

---

## Use Cases:
- Medical image filtering from large datasets
- Preliminary medical triage systems
- Dataset curation before further AI training
- Content moderation in healthcare media

---

## Results:
- **Validation Accuracy**: ~95% (on test set)
- **High generalization** on unseen test samples
- Robust to different image formats


