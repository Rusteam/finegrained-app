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
    features: List[List[float]]
    data: List[dict]


class SearchRequest(BaseModel):
    features: List[List[float]]
    top_k: Optional[int] = 3


app = FastAPI()
triton = TritonClient()
sim = SimilaritySearch()
vectors_path = Path(os.getenv("VECTOR_STORAGE", "./vectors"))
vectors_path.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def hello():
    return {"message": "Hello World"}


@app.get("/models")
async def list_models():
    return triton.list_models()


@app.get("/models/{model_name}")
async def get_model_config(model_name: str):
    return triton.get_model_details(model_name)


@app.post("/models/{model_name}/predict")
async def run_inference(model_name: str, request_body: InferRequest):
    assert (
        request_body.image or request_body.text
    ), "Either text or image have to be sent as payload"
    return triton.predict(model_name, request_body.__dict__)


def _make_vector_path(data_name: str) -> str:
    path = vectors_path / shlex.quote(data_name)
    return str(path.with_suffix(".faiss"))


@app.post("/embeddings/{data_name}")
async def add_vectors(
    data_name: str, request_body: EmbedRequest
):
    res = sim.index(vectors=request_body.features,
              data=request_body.data,
              dest=_make_vector_path(data_name))
    return res


@app.post("/embeddings/{data_name}/search")
async def search_similar(data_name: str, request_body: SearchRequest):
    res = sim.query_vectors(index_file=_make_vector_path(data_name),
                            vectors=request_body.features,
                            top_k=request_body.top_k)
    return res
