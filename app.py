# app.py
"""
Streamlit app: Nhận dạng chữ số viết tay (MNIST) bằng K-NN
Yêu cầu:
- model.joblib (chứa dict: {"model":knn, "scaler":..., "pca":...}) trong cùng thư mục
- optional: streamlit-drawable-canvas để vẽ trực tiếp
Chạy:
    streamlit run app.py
"""

import streamlit as st
from PIL import Image, ImageOps, Image
import numpy as np
import joblib
import os
import io

st.set_page_config(page_title="MNIST K-NN (Streamlit)", layout="centered")
st.title("Nhận dạng chữ số viết tay (MNIST) — K-NN")
st.markdown("Upload ảnh chữ số hoặc vẽ trực tiếp. App dùng model `model.joblib` (0-9).")

MODEL_PATH = "model.joblib"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    return bundle

bundle = load_model()

if bundle is None:
    st.warning("Không tìm thấy file `model.joblib` trong thư mục. Bạn có thể:")
    st.markdown("- Chạy script huấn luyện (ví dụ `train_knn.py`) để tạo `model.joblib`.\n- Hoặc upload file `model.joblib` bằng control dưới đây.")
    uploaded = st.file_uploader("Upload file model.joblib (tùy chọn)", type=["joblib"])
    if uploaded is not None:
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Đã upload model.joblib. Reload trang để load model.")
    st.stop()

model = bundle.get("model", None)
scaler = bundle.get("scaler", None)
pca = bundle.get("pca", None)

st.sidebar.subheader("Model info")
if model is not None:
    st.sidebar.write(f"Model: {type(model).__name__}")
    try:
        st.sidebar.write(f"n_neighbors = {model.n_neighbors}")
    except Exception:
        pass
else:
    st.sidebar.write("Chưa có model trong model.joblib")

# Option: enable drawable canvas
use_canvas = st.sidebar.checkbox("Cho phép vẽ trực tiếp (canvas)", value=True)

# Try to import canvas only if user wants it
canvas_image = None
if use_canvas:
    try:
        from streamlit_drawable_canvas import st_canvas
        st.sidebar.write("Canvas: dùng chuột (hoặc bút) để vẽ chữ số màu đen trên nền trắng.")
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,1)",
            stroke_width=18,
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result is not None and canvas_result.image_data is not None:
            # image_data is RGBA numpy array (h, w, 4)
            canvas_img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("RGB")
            st.image(canvas_img, caption="Ảnh vẽ (canvas)", width=200)
            canvas_image = canvas_img
    except Exception as e:
        st.sidebar.error("Để dùng canvas bạn cần cài `streamlit-drawable-canvas` hoặc bỏ chọn canvas.")
        # don't stop; allow file upload below

uploaded_file = st.file_uploader("Upload ảnh chữ số (PNG/JPG) — hoặc vẽ ở canvas", type=["png","jpg","jpeg"])

# Select which input to use (priority: upload > canvas)
img_input = None
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh bạn upload", width=200)
        img_input = img
    except Exception as e:
        st.error("Không thể mở file ảnh. Hãy kiểm tra file.")
elif canvas_image is not None:
    img_input = canvas_image

if img_input is None:
    st.info("Upload ảnh hoặc vẽ chữ số để dự đoán.")
    st.stop()

# Utility: get resample method compatible with Pillow versions
def get_resample_method():
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        # Pillow < 10
        return Image.ANTIALIAS

RESAMPLE = get_resample_method()

def preprocess_image(img: Image.Image, size=(28,28)):
    """
    Input: PIL Image (any size, RGB or L)
    Steps:
    - convert grayscale
    - pad to square with white background
    - resize to (28,28) with Lanczos (ANTIALIAS)
    - invert (so digit is bright like MNIST) and normalize to 0..1
    - flatten to (1, 784)
    """
    # convert to grayscale
    img = img.convert("L")
    # auto-crop surrounding empty border (optional) -> helps centering
    # but keep safe: use ImageOps.fit with centering via padding
    old_size = img.size  # (width, height)
    max_side = max(old_size)
    # create white square and paste centered
    new_img = Image.new("L", (max_side, max_side), color=255)
    paste_pos = ((max_side - old_size[0]) // 2, (max_side - old_size[1]) // 2)
    new_img.paste(img, paste_pos)
    # resize to target
    new_img = new_img.resize(size, RESAMPLE)
    arr = np.array(new_img).astype(np.float32)
    # invert: MNIST foreground is white (high) when normalized (we will invert to make foreground high)
    arr = 255.0 - arr
    # normalize 0..1
    arr = arr / 255.0
    flat = arr.reshape(1, -1)
    return flat, new_img  # return also processed PIL image for visualization

# Show processed preview and predict button
st.subheader("Ảnh tiền xử lý (28x28) và dự đoán")
col1, col2 = st.columns([1,1])
with col1:
    st.write("Ảnh gốc")
    st.image(img_input, width=200)
with col2:
    st.write("Ảnh sau tiền xử lý (28x28)")
    x_vis, proc_img = preprocess_image(img_input, size=(28,28))
    st.image(proc_img.resize((140,140), RESAMPLE), width=140)

# Predict
if st.button("Dự đoán chữ số"):
    x = x_vis  # shape (1,784)
    # apply scaler/pca same as training (robust: detect expected scale)
    try:
        # x currently in 0..1 (preprocess_image normalized)
        x_orig = x.copy()

        if scaler is not None:
            # Heuristic: if scaler.mean_ exists and its max > 1 => scaler was fit on 0..255 data
            scaler_mean = getattr(scaler, "mean_", None)
            if scaler_mean is not None and np.max(np.abs(scaler_mean)) > 1.0:
                x_for_scaler = x * 255.0
            else:
                x_for_scaler = x.copy()

            # debug prints (remove or comment out when ok)
            st.write("Input range before scaler:", float(x_for_scaler.min()), float(x_for_scaler.max()))
            st.write("Scaler mean (sample):", float(np.mean(scaler_mean)) if scaler_mean is not None else "N/A")

            x = scaler.transform(x_for_scaler)
        else:
            # if no scaler, keep as-is (0..1)
            st.write("No scaler in model bundle; using pixel range 0..1 for prediction.")

        if pca is not None:
            x = pca.transform(x)
    except Exception as e:
        st.warning("Lỗi khi áp dụng scaler/pca: kiểm tra model.joblib. Dự đoán sẽ dùng vector gốc.")
        x = x_orig

    try:
        pred = model.predict(x)[0]
        st.success(f"Dự đoán: **{int(pred)}**")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        st.stop()

    # show probabilities if available
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)[0]
            top_idx = np.argsort(probs)[::-1][:5]
            st.write("Top 5 dự đoán (label : probability):")
            for idx in top_idx:
                st.write(f"{idx} : {probs[idx]:.3f}")
        except Exception:
            pass
    else:
        # try to show neighbor distances (kneighbors)
        try:
            dists, idxs = model.kneighbors(x, n_neighbors=min(5, getattr(model, "n_neighbors", 5)))
            st.write("Khoảng cách đến 5 neighbors (nhỏ = gần):")
            st.write(np.round(dists[0], 4))
        except Exception:
            pass

    # Optional: show raw vector stats
    if st.checkbox("Hiển thị vector đầu vào (784 dim)"):
        st.write(x.flatten()[:100].tolist(), " ...")  # show first 100 dims
    st.write("Scaler present:", scaler is not None)
