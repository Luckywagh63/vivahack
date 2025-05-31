import streamlit as st
import torch

@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True
    )
    return model

model = load_model()

st.title("YOLOv5 Tree Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = uploaded_file.read()
    # save uploaded image
    with open("input.jpg", "wb") as f:
        f.write(img)

    results = model("input.jpg")
    results.save()  # saves to runs/detect/exp

    st.image("runs/detect/exp/input.jpg")
