#type: ignore
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Chuẩn hóa dữ liệu
x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.0
x_test  = x_test.reshape(-1,28,28,1).astype('float32') / 255.0

# 3. One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# 4. Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Train model
model.fit(x_train, y_train, validation_split=0.1, epochs=15, batch_size=128)

# 7. Evaluate model
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# 8. Lưu model
model.save_weights('models/mnist_cnn.weights.h5')