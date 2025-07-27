import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import json

# --- Cấu hình và Tải Mô hình/Nhãn ---
trainer_dir = 'trainer'

# Đường dẫn cho mô hình cảm xúc (CNN)
emotion_model_path = os.path.join(trainer_dir, 'emotion_model.h5')
emotion_labels_path = os.path.join(trainer_dir, 'emotion_labels.json')

# Đường dẫn cho Haar Cascade
cascade_path = "haarcascade_frontalface_default.xml"

# Tải mô hình cảm xúc (CNN)
try:
    emotion_model = load_model(emotion_model_path)
    print(f"[INFO] Đã tải mô hình cảm xúc {emotion_model_path} thành công.")
    emotion_model.summary() # In ra tóm tắt mô hình để kiểm tra input shape
except Exception as e:
    print(f"[ERROR] Không thể tải mô hình cảm xúc {emotion_model_path}. Đảm bảo bạn đã chạy 02_emotion_training.py và mô hình được lưu đúng cách: {e}")
    exit()

# Tải bản đồ nhãn cảm xúc
try:
    with open(emotion_labels_path, 'r') as f:
        emotion_labels = json.load(f)
    print(f"[INFO] Đã tải nhãn cảm xúc từ {emotion_labels_path} thành công: {emotion_labels}")
except Exception as e:
    print(f"[ERROR] Không thể tải {emotion_labels_path}. Đảm bảo bạn đã chạy 02_emotion_training.py và tệp nhãn được lưu đúng cách: {e}")
    exit()

# Tải bộ phát hiện khuôn mặt Haar Cascade
faceCascade = cv2.CascadeClassifier(cascade_path)
if faceCascade.empty():
    print(f"[ERROR] Không thể tải Haar Cascade file: {cascade_path}. Đảm bảo tệp này nằm cùng thư mục với tập lệnh.")
    exit()
else:
    print(f"[INFO] Đã tải Haar Cascade file: {cascade_path} thành công.")

font = cv2.FONT_HERSHEY_SIMPLEX

# --- Khởi tạo Camera ---
cam = cv2.VideoCapture(0) # 0 cho camera mặc định
if not cam.isOpened():
    print("[ERROR] Không thể mở camera. Đảm bảo camera đã được kết nối và không bị sử dụng bởi ứng dụng khác.")
    exit()

cam.set(3, 640) # đặt độ rộng video
cam.set(4, 480) # đặt chiều cao video

# Xác định kích thước cửa sổ tối thiểu để được nhận dạng là khuôn mặt
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\n[INFO] Đang khởi động nhận diện cảm xúc. Nhấn 'ESC' để thoát.")

# --- Vòng lặp nhận diện ---
frame_count = 0
while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Không thể đọc khung hình từ camera. Đang thoát.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1, # Điều chỉnh scaleFactor nếu cần (ví dụ: 1.05 - 1.3)
        minNeighbors=5,  # Điều chỉnh minNeighbors nếu cần (ví dụ: 3 - 6)
        minSize=(int(minW), int(minH)),
    )
    
    frame_count += 1
    if frame_count % 30 == 0: # In thông báo mỗi 30 khung hình
        print(f"[DEBUG] Đã phát hiện {len(faces)} khuôn mặt trong khung hình.")

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cắt vùng khuôn mặt
        face_roi_gray = gray[y:y+h, x:x+w] # Vùng ảnh xám cho mô hình cảm xúc

        # --- Nhận diện CẢM XÚC (sử dụng CNN) ---
        emotion_text = "Predicting..."
        emotion_confidence_text = ""
        try:
            # Tiền xử lý khuôn mặt cho mô hình CNN (PHẢI KHỚP VỚI HUẤN LUYỆN)
            processed_face_emotion = cv2.resize(face_roi_gray, (48, 48))
            processed_face_emotion = processed_face_emotion.astype('float32') / 255.0
            processed_face_emotion = np.expand_dims(processed_face_emotion, axis=0)
            processed_face_emotion = np.expand_dims(processed_face_emotion, axis=-1)

            predictions = emotion_model.predict(processed_face_emotion, verbose=0)
            
            emotion_index = np.argmax(predictions[0])
            emotion = emotion_labels[emotion_index]
            confidence = predictions[0][emotion_index] * 100

            if confidence > 30: # Ngưỡng độ tin cậy cho cảm xúc
                emotion_text = f"Emotion: {emotion}"
                emotion_confidence_text = f"Conf: {confidence:.2f}%"
            else:
                emotion_text = "Emotion: Unknown"
                emotion_confidence_text = ""

        except Exception as e:
            print(f"[ERROR] Lỗi khi dự đoán cảm xúc: {e}")
            emotion_text = "Emotion Error"
            emotion_confidence_text = ""

        # --- Hiển thị kết quả trên khung hình ---
        # Hiển thị cảm xúc
        cv2.putText(img, emotion_text, (x + 5, y - 5), font, 0.9, (0, 255, 255), 2)
        if emotion_confidence_text:
            cv2.putText(img, emotion_confidence_text, (x + 5, y + 20), font, 0.7, (0, 255, 255), 1)


    cv2.imshow('Nhận diện Cảm xúc', img)

    k = cv2.waitKey(10) & 0xff # Nhấn 'ESC' để thoát video
    if k == 27:
        break

# --- Dọn dẹp ---
print("\n[INFO] Đang thoát khỏi chương trình và dọn dọn")
cam.release()
cv2.destroyAllWindows()