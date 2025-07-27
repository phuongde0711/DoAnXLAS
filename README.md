# 😄 Nhận diện cảm xúc từ khuôn mặt sử dụng học sâu

Dự án này sử dụng OpenCV và mô hình học sâu để nhận diện biểu cảm khuôn mặt theo thời gian thực và phân loại cảm xúc như **Vui**, **Buồn**, **Tức giận**, **Ngạc nhiên**, v.v.

---

## 📁 Cấu trúc thư mục

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
```

## 🚀 How to Run

** 🔧 1. Huấn luyện mô hình
