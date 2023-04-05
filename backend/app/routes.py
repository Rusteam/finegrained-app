import numpy as np
import shlex
from pathlib import Path

import os

from typing import Optional, Union, List

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, validator, root_validator

from ..utils.image_utils import decode_base64_image
from ..utils.similarity import SimilaritySearch
from ..utils.triton import TritonClient
from ..utils.pipelines import Pipelines


REQUEST_TYPE = Union[str, List[str]]
DESCRIPTIONS = dict(
    groupby="Group results by this key and return only top matching element",
    top_k="Number of top matches to return",
    squeeze="If parent array contains only one element, then takes the first element only",
)


def _post_init_image(image: REQUEST_TYPE) -> np.ndarray:
    if image is not None:
        images = image
        if isinstance(images, str):
            images = [images]
        images = map(decode_base64_image, images)
        return list(images)
    else:
        return None


def _post_init_text(text: REQUEST_TYPE) -> np.ndarray:
    if text is not None:
        texts = text
        if isinstance(texts, str):
            texts = [texts]
        return np.array(texts)[..., np.newaxis]
    else:
        return None


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


app = FastAPI()
triton = TritonClient()
sim = SimilaritySearch()
pipe = Pipelines(triton)
vectors_path = Path(os.getenv("VECTOR_STORAGE", "./vectors"))
vectors_path.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def hello():
    """Check that the api is up and running."""
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    """List models available for inference."""
    return {"results": triton.list_models()}


@app.get("/models/{model_name}")
async def model_config(model_name: str):
    """Find out model input and output scheme."""
    return {"results": triton.get_model_details(model_name)}


@app.post("/models/{model_name}/predict")
async def predict(model_name: str, request_body: InferRequest, top_k: int = 0):
    """Run model inference with given data."""
    predicted = triton.predict(model_name, request_body.dict(), top_k=top_k)
    return {"results": predicted}


@app.post("/models/{model_name}/search/{data_name}")
async def predict_and_search(
    model_name: str,
    data_name: str,
    request_body: InferRequest,
    top_k: int = Query(default=5, description=DESCRIPTIONS["top_k"]),
    squeeze: bool = Query(default=False, description=DESCRIPTIONS["squeeze"]),
    groupby: Optional[str] = Query(default=None, description=DESCRIPTIONS["groupby"])
):
    """Extract features from raw data and compare against database."""
    prediction = triton.predict(model_name, request_body.dict())
    res = sim.query_vectors(
        index_file=_make_vector_path(data_name),
        vectors=prediction["embeddings"],
        top_k=top_k,
        groupby=groupby
    )

    if squeeze and len(res) == 1:
        res = res[0]

    return {"results": res}


def _make_vector_path(data_name: str) -> str:
    path = vectors_path / shlex.quote(data_name)
    return str(path.with_suffix(".faiss"))


@app.get("/embeddings")
async def list_data():
    data = [p.stem for p in vectors_path.glob("*.faiss")]
    return {"results": data}


@app.post("/embeddings/{data_name}")
async def index_data(data_name: str, request_body: EmbedRequest):
    """Ingest vectors and data into a database for searching."""
    assert len(request_body.vectors) == len(
        request_body.data
    ), f"Data and vectors have to be of the same length"
    res = sim.index(
        vectors=request_body.vectors,
        data=request_body.data,
        dest=_make_vector_path(data_name),
    )
    return {"results": res}


@app.post("/embeddings/{data_name}/search")
async def search_similar(data_name: str, request_body: SearchRequest):
    """Find top matching samples in a database"""
    res = sim.query_vectors(
        index_file=_make_vector_path(data_name),
        vectors=request_body.vectors,
        top_k=request_body.top_k,
        groupby=request_body.groupby
    )
    return {"results": res}


@app.get("/pipelines")
async def list_pipelines():
    """List available model pipelines."""
    return dict(results=pipe.list())


@app.get("/pipelines/{pipeline}")
async def pipeline_config(pipeline: str):
    one = pipe(pipeline)
    return dict(results=one.get_config())


@app.post("/pipelines/{pipeline}/predict")
async def run_pipeline(
    pipeline: str,
    request_body: InferRequest,
    model: List[str] = Query(default=...),
):
    """Run inference pipeline with requested models."""
    infer_pipe = pipe(pipeline, model_names=model)
    results = infer_pipe.run(request_body.dict())
    return dict(results=results)
