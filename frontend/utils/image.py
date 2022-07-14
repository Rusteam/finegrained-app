"""Helper functions to deal with image data.
"""
import base64

from io import BytesIO

from typing import List, Dict, Union

import numpy as np
from PIL import ImageDraw, ImageFont, Image


def xywh2xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on
    its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin,
    xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 36)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the
    # bounding
    # box exceeds the top of the image, stack the strings below the bounding
    # box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_detections(image: np.ndarray, detections: List[Dict]) -> Image:
    """Draw bounding boxes and labels on images

    Args:
        image: source image to draw on
        detections: a list of detections as dict with
            (x_c,y_c,w,h) "bounding_box" field and a "label" field

    Returns:
        Image obj with drawings
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    for one in detections:
        xmin, ymin, xmax, ymax = xywh2xyxy(*one["bounding_box"])

        text = f"{one['label']} ({one['score']:.2f})"
        draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
                                   display_str_list=[text])
    return image


def encode_image_base64(image: Image) -> str:
    img_file = BytesIO()
    image.save(img_file, format="JPEG")
    img_bytes = img_file.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64.decode()
