# 匯入必要的庫
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# 下載並加載MNIST數據集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 對圖片進行預處理以用於CNN
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255

# 建立CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 使用Canny邊緣檢測演算法
def canny_edge_detector(image_path):
    image = cv2.imread(image_path, 0)
    edges_detected = cv2.Canny(image, 100, 200)
    cv2.imshow('Edges', edges_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges_detected

# 預測函數
def predict_digit(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉為灰度圖
    img = cv2.resize(img, (28, 28))  # 調整圖片大小以匹配訓練數據
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255
    prediction = model.predict(img)
    return np.argmax(prediction)

# 用於預測的示例圖片路徑
image_path = 'path_to_your_image.jpg'  # 替換為你的圖片路徑

# 執行Canny邊緣檢測
edges = canny_edge_detector(image_path)

# 預測數字
img = cv2.imread(image_path)
predicted_digit = predict_digit(img)
print(f'Predicted digit: {predicted_digit}')
