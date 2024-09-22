# 必要的模組導入
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Step 1: 從MNIST數據集加載數據
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

# Step 2: 使用OpenCV進行邊緣檢測
def apply_canny(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    return edges

# 將Canny邊緣檢測應用於訓練和測試數據集
mnist_train_edges = np.array([apply_canny(img) for img in mnist_train_images])
mnist_test_edges = np.array([apply_canny(img) for img in mnist_test_images])

# Step 3: 擴張處理
# 這一步通常涉及到使用膨脹等操作增強圖像特徵，但這裡省略了實際的膨脹步驟

# Step 4: 設計一個簡單的卷積神經網絡 (CNN)
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: 使用設計好的CNN進行MNIST數據的訓練和預測
# 此步驟需要實際執行訓練過程，這在這裡無法完成，因此我們將展示如何進行訓練
# model.fit(mnist_train_edges, to_categorical(mnist_train_labels), epochs=5, batch_size=32, validation_split=0.1)

# 模型摘要
model.summary()
