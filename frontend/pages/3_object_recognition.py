"""Object detection streamlit page.
"""
import numpy as np
from PIL import Image
import streamlit as st

from utils.client import Client
from utils.image import encode_image_base64, draw_detections
from utils.st import set_page

client = Client()

set_page(
    "Finegrained.AI - Object recognition pipeline",
    "Detect objects on an image cand classify each object.",
)

models = client.list_models()
detector = st.selectbox(label="Select a detector", options=models)
classifier = st.selectbox(label="Select a classifier", options=models)

image_file = st.file_uploader(label="Upload an image")


_get_max_id = lambda x: np.argmax(x)


def _parse_detections(detections: dict) -> list:
    detections = [
        {
            "bounding_box": d["box"],
            "label": d["label"],
            "score": d["confidence"],
        }
        for d in detections
    ]
    return detections


if image_file is not None:
    try:
        img = Image.open(image_file)
        detections = client.run_pipeline(
            "object-recognition",
            data={"image": encode_image_base64(img)},
            params={"model": [detector, classifier]},
        )
        detections = _parse_detections(detections)
        draw_detections(img, detections)
        st.image(img)

    except Exception as e:
        st.error(e.args)
