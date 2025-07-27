# 😄 Nhận diện cảm xúc từ khuôn mặt sử dụng học sâu

Dự án này sử dụng OpenCV và mô hình học sâu để nhận diện biểu cảm khuôn mặt theo thời gian thực và phân loại cảm xúc như **Vui**, **Buồn**, **Tức giận**, **Ngạc nhiên**, v.v.

---

## 📁 Cấu trúc thư mục

```
Emotion-detection/
├── haarcascade_frontalface_default.xml   # Mô hình Haar Cascade phát hiện khuôn mặt
├── recognition.py                        # Mã chạy nhận diện cảm xúc theo thời gian thực
├── training.py                           # Mã huấn luyện mô hình cảm xúc
├── train/                                # Dữ liệu huấn luyện
├── test/                                 # Dữ liệu kiểm thử
├── trainer/                              # Mô hình đã huấn luyện
└── README.md                             # Tệp mô tả dự án
```
---

## 🛠️ Yêu cầu cài đặt

- Python 3.x
- OpenCV
- Pillow
- NumPy
- TensorFlow / Keras *(nếu bạn sử dụng trong `training.py`)*

Cài đặt thư viện:

```bash
pip install opencv-python pillow numpy
```

---

## 🚀 Cách chạy chương trình
🔧 1. Huấn luyện mô hình
Bước 1: Đặt ảnh khuôn mặt vào thư mục train/, mỗi thư mục con là một loại cảm xúc 
Bước 2: Chạy lệnh huấn luyện
```bash
python training.py
```
🎥 2. Chạy chương trình nhận diện cảm xúc thời gian thực
Kết nối webcam và chạy:
```bash
python recognition.py
```

---

## 🔍Tính năng
- 📷 Phát hiện khuôn mặt theo thời gian thực bằng Haar Cascade
- 😊 Nhận diện cảm xúc từ biểu cảm khuôn mặt
- ⚙️ Dễ tùy biến và mở rộng mô hình

---

## 📸 Ví dụ kết quả



