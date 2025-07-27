# Emotion Detection from Facial Expressions using Deep Learning

This project detects human emotions from facial expressions in real-time using OpenCV and a deep learning model. The system first detects faces using Haar Cascade, then classifies emotions such as **Happy**, **Sad**, **Angry**, **Surprised**, etc.

---

## 📦 Folder Structure

Emotion-detection/
│
├── haarcascade_frontalface_default.xml # Haar Cascade model for face detection
├── recognition.py # Main script for real-time emotion detection
├── training.py # Script to train emotion classifier
├── train/ # Training dataset
├── test/ # Testing dataset
├── trainer/ # Saved trained model
└── README.md # This file


---

## 🛠️ Requirements

- Python 3.x
- OpenCV
- Pillow
- NumPy
- TensorFlow / Keras *(if used inside `training.py`)*

Install dependencies:

```bash
pip install opencv-python pillow numpy
