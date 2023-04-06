from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator, root_validator

from ..utils.image_utils import decode_base64_image

REQUEST_TYPE = str | List[str]
DESCRIPTIONS = dict(
    groupby="Group results by this key and return only top matching element",
    top_k="Number of top matches to return",
    squeeze="If parent array contains only one element, then takes the first element only",
)


def _post_init_image(image: REQUEST_TYPE) -> np.ndarray | list[np.ndarray] | None:
    if image is not None:
        images = image
        if isinstance(images, str):
            images = [images]
        images = map(decode_base64_image, images)
        return list(images)
    else:
        return None


def _post_init_text(text: REQUEST_TYPE) -> np.ndarray | None:
    if text is not None:
        texts = text
        if isinstance(texts, str):
            texts = [texts]
        return np.array(texts)[..., np.newaxis]
    else:
        return None


class EmbedRequest(BaseModel):
    vectors: List[List[float]] = Field(
        default=..., description="Vector representation of input data"
    )
    data: List[dict] = Field(
        default=...,
        description="Data fields that will be returned when searching",
    )


class SearchRequest(BaseModel):
    vectors: List[List[float]] = Field(
        default=..., description="Vector representation of input data"
    )
    top_k: Optional[int] = Field(
        default=3, description=DESCRIPTIONS["top_k"]
    )
    groupby: Optional[str] = Field(default=..., description=DESCRIPTIONS["groupby"])


class InferRequest(BaseModel):
    text: Optional[REQUEST_TYPE] = Field(
        default=None, description="Plain text"
    )
    image: Optional[REQUEST_TYPE] = Field(
        default=None, description="Base64 encoded image(s)"
    )

    @validator("image")
    def decode_base64_image(cls, v):
        return _post_init_image(v)

    @validator("text")
    def stack_texts_array(cls, v):
        return _post_init_text(v)

    @root_validator
    def not_all_none(cls, values):
        assert any(
            [values.get(v) is not None for v in ["text", "image"]]
        ), "Either text or image have to be sent as payload"
        return values
