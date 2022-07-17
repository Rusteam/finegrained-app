"""Object detection streamlit page.
"""
import numpy as np
from PIL import Image
import streamlit as st

from utils.client import Client
from utils.image import encode_image_base64, draw_detections

client = Client()

model = st.selectbox(label="Select a model", options=client.list_models())
image_file = st.file_uploader(label="Upload an image")


_get_max_id = lambda x: np.argmax(x)


if image_file is not None:
    img = Image.open(image_file)
    top_classes = client.predict(model_name=model,
                                 data={"image": encode_image_base64(img)},
                                 params=dict(top_k=5))
    st.image(img)
    st.write(top_classes)
