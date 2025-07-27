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

<img width="216" height="228" alt="image" src="https://github.com/user-attachments/assets/7b9fb77f-e0a7-4b5b-bc01-d144007cd015" />

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
<img width="793" height="594" alt="image" src="https://github.com/user-attachments/assets/90a43065-6c73-4881-8c74-70e94171f393" />
<img width="798" height="599" alt="image" src="https://github.com/user-attachments/assets/932caeab-4702-4183-9a24-13ada8402730" />
<img width="793" height="598" alt="image" src="https://github.com/user-attachments/assets/6aed2797-6c7f-4558-b3c2-c738ccdb23b9" />




