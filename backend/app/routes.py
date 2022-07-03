import shlex

from pathlib import Path

import os

from typing import Optional, Union, List

from fastapi import FastAPI
from pydantic import BaseModel

from ..utils.similarity import SimilaritySearch
from ..utils.triton import TritonClient


REQUEST_TYPE = Union[str, List[str]]


class InferRequest(BaseModel):
    text: Optional[REQUEST_TYPE] = None
    image: Optional[REQUEST_TYPE] = None


class EmbedRequest(BaseModel):
    vectors: List[List[float]]
    data: List[dict]


class SearchRequest(BaseModel):
    vectors: List[List[float]]
    top_k: Optional[int] = 3


app = FastAPI()
triton = TritonClient()
sim = SimilaritySearch()
vectors_path = Path(os.getenv("VECTOR_STORAGE", "./vectors"))
vectors_path.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def hello():
    """Check that the api is up and running."""
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    """List models available for inference."""
    return triton.list_models()


@app.get("/models/{model_name}")
async def model_config(model_name: str):
    """Find out model input and output scheme."""
    return triton.get_model_details(model_name)


@app.post("/models/{model_name}/predict")
async def predict(model_name: str, request_body: InferRequest):
    """Run model inference with given data."""
    assert (
        request_body.image or request_body.text
    ), "Either text or image have to be sent as payload"
    return triton.predict(model_name, request_body.dict())


@app.post("/models/{model_name}/search/{data_name}")
async def predict_and_search(
    model_name: str,
    data_name: str,
    request_body: InferRequest,
    top_k: int = 5,
    squeeze: bool = False,
):
    """Extract features from raw data and compare against database."""
    assert (
        request_body.image or request_body.text
    ), "Either text or image have to be sent as payload"
    prediction = triton.predict(model_name, request_body.dict())
    res = sim.query_vectors(
        index_file=_make_vector_path(data_name),
        vectors=prediction["embeddings"],
        top_k=top_k,
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
    return res


@app.post("/embeddings/{data_name}/search")
async def search_similar(data_name: str, request_body: SearchRequest):
    """Find top matching samples in a database"""
    res = sim.query_vectors(
        index_file=_make_vector_path(data_name),
        vectors=request_body.vectors,
        top_k=request_body.top_k,
    )
    return res
