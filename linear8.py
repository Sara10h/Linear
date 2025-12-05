import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Image Processing Tools", layout="wide")


st.title("Image Processing Tools")

# =========================
# UPLOAD IMAGE
# =========================
uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.warning("Please upload an image.")
    st.stop()

img = Image.open(uploaded)
img_arr = np.array(img)

# =========================
# SESSION STATE FOR RESET
# =========================
if "result" not in st.session_state:
    st.session_state.result = img_arr.copy()

# =========================
# SIDEBAR - FILTER SETTINGS
# =========================
st.sidebar.subheader("Transformations")

# Translation
use_translation = st.sidebar.checkbox("Translation")
tx, ty = 0, 0
if use_translation:
    tx = st.sidebar.slider("Shift X", -200, 200, 0)
    ty = st.sidebar.slider("Shift Y", -200, 200, 0)

# Scaling
use_scaling = st.sidebar.checkbox("Scaling")
scale = 1.0
if use_scaling:
    scale = st.sidebar.slider("Scale Factor", 0.1, 3.0, 1.0)

# Rotation
use_rotation = st.sidebar.checkbox("Rotation")
angle = 0
if use_rotation:
    angle = st.sidebar.slider("Rotation Angle (°)", 0, 360, 0)

# Shearing
use_shearing = st.sidebar.checkbox("Shearing")
shx, shy = 0.0, 0.0
if use_shearing:
    shx = st.sidebar.slider("Shear X", -1.0, 1.0, 0.0)
    shy = st.sidebar.slider("Shear Y", -1.0, 1.0, 0.0)

# Reflection
use_reflection = st.sidebar.checkbox("Reflection")
axis = "Horizontal"
if use_reflection:
    axis = st.sidebar.selectbox("Reflect Axis", ["Horizontal", "Vertical"])

# Blur
use_blur = st.sidebar.checkbox("Blur")
blur_level = 1
if use_blur:
    blur_level = st.sidebar.slider("Blur Level", 1, 25, 3, step=2)

# Sharpen
use_sharpen = st.sidebar.checkbox("Sharpen")
sharp_level = 1
color_mode = "All"
if use_sharpen:
    sharp_level = st.sidebar.slider("Sharpen Level", 1, 10, 1)
    color_mode = st.sidebar.selectbox("Sharpen Color Channel", ["All", "Red", "Green", "Blue"])

# =========================
# APPLY ALL FILTERS
# =========================
result = img_arr.copy()

# Translation
if use_translation:
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

# Scaling
if use_scaling:
    result = cv2.resize(result, None, fx=scale, fy=scale)

# Rotation
if use_rotation:
    h, w = result.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    result = cv2.warpAffine(result, M, (w, h))

# Shearing
if use_shearing:
    M = np.float32([[1, shx, 0],
                    [shy, 1, 0]])
    result = cv2.warpAffine(result, M, (result.shape[1], result.shape[0]))

# Reflection
if use_reflection:
    if axis == "Horizontal":
        result = cv2.flip(result, 0)
    else:
        result = cv2.flip(result, 1)

# Blur
if use_blur:
    result = cv2.GaussianBlur(result, (blur_level, blur_level), 0)

# Sharpen
if use_sharpen:
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + sharp_level, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(result, -1, kernel)

    if color_mode == "Red":
        result[:, :, 0] = sharpened[:, :, 0]
    elif color_mode == "Green":
        result[:, :, 1] = sharpened[:, :, 1]
    elif color_mode == "Blue":
        result[:, :, 2] = sharpened[:, :, 2]
    else:
        result = sharpened

st.session_state.result = result

# =========================
# DISPLAY IMAGE
# =========================
col1, col2 = st.columns(2)
col1.image(img_arr, caption="Original Image", width="stretch")
col2.image(st.session_state.result, caption="Result", width="stretch")

# =========================
# DOWNLOAD BUTTON
# =========================
st.subheader("Download Result")
result_pil = Image.fromarray(st.session_state.result)
buf = io.BytesIO()
result_pil.save(buf, format="PNG")

st.download_button(
    label="Download Image",
    data=buf.getvalue(),
    file_name="hasil_edit.png",
    mime="image/png"
)

