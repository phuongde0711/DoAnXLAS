import cv2
import numpy as np
import os
# import pandas as pd # Không cần pandas nữa vì không đọc CSV
import json # ĐÃ THÊM: Nhập thư viện json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split # Không cần train_test_split nữa vì dữ liệu đã được chia
from PIL import Image # Để đọc hình ảnh

# Đường dẫn đến thư mục gốc của tập dữ liệu FER-2013 (chứa 'train' và 'test')
dataset_root_path = '.' # Giả sử thư mục 'train' và 'test' nằm cùng cấp với tập lệnh này
train_data_path = os.path.join(dataset_root_path, 'train')
test_data_path = os.path.join(dataset_root_path, 'test')

# Đường dẫn để lưu mô hình đã huấn luyện
trainer_dir = 'trainer'

# Đảm bảo thư mục trainer tồn tại
if not os.path.exists(trainer_dir):
    os.makedirs(trainer_dir)
    print(f"[INFO] Đã tạo thư mục: {trainer_dir}")

# Định nghĩa các nhãn cảm xúc và ánh xạ chúng với số nguyên
# Đây là các nhãn tiêu chuẩn của tập dữ liệu FER-2013
emotion_labels = ["Tức giận", "disgust", "Sợ hãi", "Vui vẻ", "Bình thương", "Buồn", "Bất ngờ"]
num_classes = len(emotion_labels)
label_map = {label: i for i, label in enumerate(emotion_labels)}

# Lưu bản đồ nhãn cảm xúc vào một tệp JSON để sử dụng trong quá trình nhận diện
with open(os.path.join(trainer_dir, 'emotion_labels.json'), 'w') as f:
    json.dump(emotion_labels, f)
print(f"[INFO] Nhãn cảm xúc đã được lưu dưới dạng {os.path.join(trainer_dir, 'emotion_labels.json')}")

# Chức năng để tải dữ liệu từ các thư mục con
def load_images_from_folders(base_path, label_map, img_size=(48, 48)):
    images = []
    labels = []
    print(f"[INFO] Đang tải dữ liệu từ {base_path}...")
    for emotion_name in os.listdir(base_path):
        emotion_folder_path = os.path.join(base_path, emotion_name)
        if os.path.isdir(emotion_folder_path) and emotion_name in label_map:
            emotion_id = label_map[emotion_name]
            for filename in os.listdir(emotion_folder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')): # Hỗ trợ nhiều định dạng ảnh
                    img_path = os.path.join(emotion_folder_path, filename)
                    try:
                        # Đọc ảnh, chuyển sang thang độ xám và thay đổi kích thước
                        img = Image.open(img_path).convert('L') # Convert to grayscale
                        img = img.resize(img_size)
                        img_array = np.array(img, dtype='float32') # Sử dụng float32 cho chuẩn hóa
                        images.append(img_array)
                        labels.append(emotion_id)
                    except Exception as e:
                        print(f"Lỗi khi tải hoặc xử lý hình ảnh {img_path}: {e}")
        else:
            if os.path.isdir(emotion_folder_path):
                print(f"[CẢNH BÁO] Thư mục '{emotion_name}' trong '{base_path}' không khớp với nhãn cảm xúc đã định nghĩa. Bỏ qua.")
    
    if not images:
        print(f"[LỖI] Không tìm thấy hình ảnh nào trong thư mục '{base_path}'. Kiểm tra đường dẫn và cấu trúc thư mục.")
        return np.array([]), np.array([]) # Trả về mảng rỗng nếu không có ảnh

    return np.array(images), np.array(labels)

# Tải dữ liệu huấn luyện và kiểm tra
try:
    X_train_raw, y_train_raw = load_images_from_folders(train_data_path, label_map)
    X_test_raw, y_test_raw = load_images_from_folders(test_data_path, label_map)
except Exception as e:
    print(f"[ERROR] Lỗi khi tải dữ liệu từ thư mục: {e}")
    print("Vui lòng đảm bảo thư mục 'train' và 'test' nằm cùng cấp với tập lệnh này và chứa các thư mục con cảm xúc.")
    exit()

if X_train_raw.size == 0 or X_test_raw.size == 0:
    print("[ERROR] Không đủ dữ liệu để huấn luyện. Vui lòng kiểm tra lại tập dữ liệu FER-2013 của bạn.")
    exit()

# Chuẩn hóa pixel và mở rộng chiều cho CNN
# Giá trị pixel cần nằm trong khoảng [0, 1]
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

# Thêm chiều kênh (1 cho ảnh xám)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Chuyển đổi nhãn thành one-hot encoding
y_train = to_categorical(y_train_raw, num_classes=num_classes)
y_test = to_categorical(y_test_raw, num_classes=num_classes)

print(f"\n[INFO] Đã chuẩn bị {len(X_train)} mẫu cho huấn luyện, {len(X_test)} mẫu cho kiểm tra.")

# Xây dựng mô hình CNN (kiến trúc cải tiến)
model = Sequential([
    # Lớp tích chập đầu tiên
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Lớp tích chập thứ hai
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Lớp tích chập thứ ba
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Làm phẳng đầu ra để đưa vào các lớp Dense
    Flatten(),
    
    # Lớp Dense với Dropout để tránh overfitting
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    # Lớp đầu ra với số lượng nơ-ron bằng số lớp cảm xúc và hàm kích hoạt softmax
    Dense(num_classes, activation='softmax')
])

# Biên dịch mô hình
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Định nghĩa Callbacks để cải thiện quá trình huấn luyện
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001)

# Huấn luyện mô hình
print("\n[INFO] Đang huấn luyện mô hình. Quá trình này có thể mất một thời gian (vài phút đến vài giờ tùy cấu hình máy)...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test), # Sử dụng X_test, y_test làm tập validation
    callbacks=[early_stopping, reduce_lr]
)

# Lưu mô hình đã huấn luyện
model_save_path = os.path.join(trainer_dir, 'emotion_model.h5')
model.save(model_save_path)
print(f"\n[INFO] Mô hình cảm xúc đã được lưu dưới dạng {model_save_path}")

# Đánh giá mô hình trên tập kiểm tra cuối cùng
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[INFO] Độ chính xác trên tập kiểm tra cuối cùng: {accuracy*100:.2f}%")
print("\n[INFO] Hoàn tất huấn luyện mô hình. Thoát khỏi chương trình.")
