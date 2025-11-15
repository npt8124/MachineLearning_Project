# app.py
#type: ignore
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import cv2

# ================= Model architecture =================
def build_cnn_model():
    # Tạo model mới **mỗi lần gọi**
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# ================= Load model =================
model = build_cnn_model()
model.load_weights('models/mnist_cnn.weights.h5')  # file train_cnn.py tạo ra
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ================= Streamlit UI =================
st.title("Nhận dạng nhiều chữ số MNIST")
st.write("Chọn chế độ: vẽ canvas hoặc upload ảnh")

mode = st.radio("Chọn chế độ", ["Vẽ canvas", "Upload ảnh"])

# ================= Hàm detect nhiều chữ số + bounding box =================
def detect_and_predict(image):
    img_cv = np.array(image.convert('L'))
    img_cv = cv2.bitwise_not(img_cv)  # đen thành trắng, trắng thành đen
    _, thresh = cv2.threshold(img_cv, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = np.array(image.convert('RGB'))
    predictions = []

    # Sắp xếp contours theo trục x để đọc từ trái sang phải
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        roi = img_cv[y:y+h, x:x+w]
        roi_img = Image.fromarray(roi).resize((28,28))
        roi_array = np.array(roi_img)/255.0
        roi_array = roi_array.reshape(1,28,28,1)
        pred = model.predict(roi_array, verbose=0)[0]
        pred_label = pred.argmax()
        predictions.append((x, y, w, h, pred, pred_label))
        cv2.rectangle(output_img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(output_img, str(pred_label), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return output_img, predictions

# ================= Chế độ Canvas =================
if mode == "Vẽ canvas":
    canvas_result = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=15,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        output_img, predictions = detect_and_predict(img)
        st.image(output_img, caption="Kết quả với bounding box", width=400)

# ================= Chế độ Upload =================
elif mode == "Upload ảnh":
    uploaded_file = st.file_uploader("Upload ảnh (png/jpg/jpeg)", type=['png','jpg','jpeg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        output_img, predictions = detect_and_predict(img)
        st.image(output_img, caption="Kết quả với bounding box", width=800)

# ================= Hiển thị xác suất =================
if 'predictions' in locals() and predictions:
    st.subheader("Xác suất dự đoán từng chữ số (từ trái sang phải)")
    for x, y, w, h, pred, label in predictions:
        sorted_indices = pred.argsort()[::-1]
        st.write(f"Bounding box ({x},{y},{w},{h}) - Nhãn dự đoán: {label}")
        for i in sorted_indices:
            st.write(f"Số {i}: {pred[i]*100:.2f}%")
        st.write("---")
