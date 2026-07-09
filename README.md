# 🫁 AI-Based Pneumonia Detection from Chest X-ray Images

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-Web%20App-orange?style=for-the-badge)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95%25-brightgreen?style=for-the-badge)

</p>

---

## 📌 Project Overview

This project presents an **AI-powered Pneumonia Detection System** that classifies Chest X-ray images as **NORMAL** or **PNEUMONIA** using **Deep Learning**.

The model is built using **Transfer Learning** with **MobileNetV2** and implemented using **PyTorch**. A simple and interactive **Gradio web interface** allows users to upload Chest X-ray images and receive predictions with confidence scores.

This project demonstrates an end-to-end deep learning workflow including:

- Data preprocessing
- Transfer Learning
- Model training
- Model evaluation
- Performance visualization
- Model deployment
- Web application development

---

# 📷 Application Demo

<p align="center">

<img src="images/demo.png" width="900">

</p>

---

# 🎯 Objectives

The primary objective of this project is to assist in the early detection of pneumonia from Chest X-ray images using Artificial Intelligence.

The project aims to:

- Detect Pneumonia from Chest X-rays
- Reduce manual diagnosis effort
- Demonstrate practical Deep Learning deployment
- Build a complete AI application using PyTorch

---

# 🧠 Model Architecture

This project uses **Transfer Learning** with **MobileNetV2**.

Architecture:

```
Chest X-ray Image
        │
        ▼
Resize (224 × 224)
        │
Normalization
        │
        ▼
MobileNetV2 Feature Extractor
        │
Adaptive Average Pooling
        │
Flatten
        │
Dropout (0.3)
        │
Linear (1280 → 64)
        │
ReLU
        │
Dropout (0.2)
        │
Linear (64 → 1)
        │
Sigmoid
        │
Prediction
```

---

# 📂 Dataset

Dataset used:

**Chest X-Ray Images (Pneumonia)**

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset contains:

- Normal Chest X-rays
- Pneumonia Chest X-rays

Directory structure:

```
train/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/

val/
    NORMAL/
    PNEUMONIA/
```

---

# ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

- Resize images to **224 × 224**
- Convert images to RGB
- Normalize using ImageNet statistics

```python
transforms.Resize((224,224))
transforms.ToTensor()
transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)
```

---

# 🏋️ Model Training

Transfer Learning:

✅ MobileNetV2 pretrained on ImageNet

Loss Function:

- Binary Cross Entropy (BCELoss)

Optimizer:

- Adam Optimizer

Epochs:

- 6

Batch Size:

- 16

---

# 📊 Model Performance

## Test Accuracy

**95%**

### Classification Report

```
              precision    recall    f1-score

NORMAL          0.93       0.94       0.93

PNEUMONIA       0.96       0.96       0.96

Overall Accuracy = 95%
```

---

## Confusion Matrix

```
[[219  15]
 [ 16 374]]
```

Interpretation:

- True Normal = 219
- False Positive = 15
- False Negative = 16
- True Pneumonia = 374

---

# 🚀 Features

✅ AI-based Pneumonia Detection

✅ Deep Learning using PyTorch

✅ MobileNetV2 Transfer Learning

✅ Confidence Score Prediction

✅ Interactive Gradio Interface

✅ Easy-to-use Web Application

---

# 💻 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| PyTorch | Deep Learning |
| Torchvision | Pretrained Models |
| MobileNetV2 | Feature Extraction |
| Gradio | Web Interface |
| PIL | Image Processing |
| NumPy | Numerical Computation |
| Matplotlib | Visualization |

---

# 📁 Project Structure

```
Pneumonia-Detection/

│

├── app.py
├── model.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md

│

├── models/
│      pneumonia_pt_best.pth

│

├── notebook/
│      pneumonia_detection.ipynb

│

├── images/
│      demo.png

│

├── examples/
│      normal.jpeg
│      pneumonia.jpeg

│

└── sample_images/
       test.jpeg
```

---

# ▶️ Installation

Clone repository

```bash
git clone https://github.com/Mohit7089/Pneumonia-Detection.git
```

Move inside project

```bash
cd Pneumonia-Detection
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
python app.py
```

Open browser

```
http://127.0.0.1:7860
```

---

# 🖥️ Using the Application

1. Launch the application.
2. Upload a Chest X-ray image.
3. Click **Submit**.
4. View the predicted class.
5. Observe confidence scores for both classes.

---

# 📈 Future Improvements

- Multi-class lung disease classification
- Grad-CAM visualization
- Model explainability
- Cloud deployment
- Docker support
- REST API integration
- Improved model accuracy
- Mobile application support

---

# 👨‍💻 Developer

**Mohit Soni**

Master of Computer Applications (MCA)

National Institute of Technology Kurukshetra

GitHub:

https://github.com/Mohit7089

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

It motivates further development and helps others discover the project.

---

# 📜 Disclaimer

This project is developed **for educational and research purposes only**.

It is **not intended to replace professional medical diagnosis** or clinical decision-making.