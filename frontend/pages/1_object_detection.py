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


def _parse_detections(detections: dict) -> list:
    detections = [{
        "bounding_box": detections["boxes"][i],
        "label": _get_max_id(detections["class_probs"][i]),
        "score": detections["scores"][i][0],
    } for i in range(len(detections["boxes"]))]
    return detections


if image_file is not None:
    img = Image.open(image_file)
    detections = client.predict(model_name=model,
                                data={"image": encode_image_base64(img)})
    detections = _parse_detections(detections)
    draw_detections(img, detections)
    st.image(img)
