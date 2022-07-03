"""Image utilities
"""
import base64

import numpy as np
from PIL import Image
from io import BytesIO


def decode_raw_image(raw_content):
    """from response.content get numpy image"""
    img = Image.open(BytesIO(raw_content))
    return np.array(img)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """decode base64 encoded image from request.data"""
    raw = base64.b64decode(base64_string)
    image = decode_raw_image(raw)
    return image


def encode_image_base64(image: Image) -> str:
    img_file = BytesIO()
    image.save(img_file, format="JPEG")
    img_bytes = img_file.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64


def load_base64_image(filepath: str) -> str:
    with open(filepath, "rb") as f:
        data = f.read()
    data_b64 = base64.b64encode(data).decode()
    return data_b64
