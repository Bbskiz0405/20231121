import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

# 載入MNIST數據集
(x_train, y_train), (_, _) = mnist.load_data()

# 顯示第一張圖片
first_image = x_train[0]
cv2.imshow('Original Image', first_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用Canny邊緣檢測
edges = cv2.Canny(first_image, 100, 200)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 正規化圖片數據並將標籤進行one-hot編碼
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(y_train)

# 建立CNN模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=64)

# 預測一張手寫數字圖片的結果
image_path = 'path_to_your_image.png'  # 替換成你的圖片路徑
img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array / 255, axis=0)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f'預測的數字為: {predicted_class}')

# 顯示原始圖片
plt.imshow(img, cmap='gray')
plt.title(f'Predicted Class: {predicted_class}')
plt.show()