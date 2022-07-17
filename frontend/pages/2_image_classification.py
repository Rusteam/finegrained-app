"""Image classification page.
"""
from PIL import Image
import streamlit as st

from utils.client import Client
from utils.image import encode_image_base64
from utils.st import display_labels, set_page

client = Client()

set_page(
    "Finegrained.AI - Image classification",
    "Label an image with top_k class labels",
)

l, r = st.columns([2, 1])
with l:
    model = st.selectbox(label="Select a model", options=client.list_models())
with r:
    top_k = st.number_input(
        label="No. of top K classes", min_value=1, max_value=10, value=5
    )
image_file = st.file_uploader(label="Upload an image")

if image_file is not None:
    try:
        img = Image.open(image_file)
        top_classes = client.predict(
            model_name=model,
            data={"image": encode_image_base64(img)},
            params=dict(top_k=top_k),
        )
        left, right = st.columns([2, 1])
        with left:
            st.image(img)
        with right:
            st.markdown("##### Results")
            display_labels(top_classes["class_probs"][0])
    except Exception as e:
        st.error(e.args)