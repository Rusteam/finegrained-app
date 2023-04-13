import shlex
from pathlib import Path

import os

from typing import Optional, List

from fastapi import FastAPI, Query, HTTPException

from . import scheme
from ..utils.similarity import SimilaritySearch
from ..utils.triton import TritonClient
from ..utils.pipelines import Pipelines
from ..utils import tracking

app = FastAPI()
triton = TritonClient()
sim = SimilaritySearch()
pipe = Pipelines(triton)
vectors_path = Path(os.getenv("VECTOR_STORAGE", "./vectors"))
vectors_path.mkdir(parents=True, exist_ok=True)


@app.get("/", tags=["health"])
async def hello():
    """Check that the api is up and running."""
    return {"message": "Hello World"}


### Models ###


@app.get("/models", tags=["models"])
async def list_models():
    """List models available for inference."""
    return {"results": triton.list_models()}


@app.get("/models/{model_name}", tags=["models"])
async def model_config(model_name: str):
    """Find out model input and output scheme."""
    return {"results": triton.get_model_details(model_name)}


@app.post("/models/{model_name}/predict", tags=["models"])
async def predict(model_name: str, request_body: scheme.InferRequest, top_k: int = 0):
    """Run model inference with given data."""
    predicted = triton.predict(model_name, request_body.dict(), top_k=top_k)
    return {"results": predicted}


### Similarity Search ###

@app.post("/models/{model_name}/search/{data_name}", tags=["similarity_search"])
async def predict_and_search(
        model_name: str,
        data_name: str,
        request_body: scheme.InferRequest,
        top_k: int = Query(default=5, description=scheme.DESCRIPTIONS["top_k"]),
        squeeze: bool = Query(default=False, description=scheme.DESCRIPTIONS["squeeze"]),
        groupby: Optional[str] = Query(default=None, description=scheme.DESCRIPTIONS["groupby"])
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


@app.get("/embeddings", tags=["similarity_search"])
async def list_data():
    data = [p.stem for p in vectors_path.glob("*.faiss")]
    return {"results": data}


@app.post("/embeddings/{data_name}", tags=["similarity_search"])
async def index_data(data_name: str, request_body: scheme.EmbedRequest):
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


@app.post("/embeddings/{data_name}/search", tags=["similarity_search"])
async def search_similar(data_name: str, request_body: scheme.SearchRequest):
    """Find top matching samples in a database"""
    res = sim.query_vectors(
        index_file=_make_vector_path(data_name),
        vectors=request_body.vectors,
        top_k=request_body.top_k,
        groupby=request_body.groupby
    )
    return {"results": res}


### Pipelines ###

@app.get("/pipelines", tags=["pipelines"])
async def list_pipelines():
    """List available model pipelines."""
    return dict(results=pipe.list())


@app.get("/pipelines/{pipeline}", tags=["pipelines"])
async def pipeline_config(pipeline: str):
    one = pipe(pipeline)
    return dict(results=one.get_config())


@app.post("/pipelines/{pipeline}/predict", tags=["pipelines"])
async def run_pipeline(
        pipeline: str,
        request_body: scheme.InferRequest,
        model: List[str] = Query(default=...),
):
    """Run inference pipeline with requested models."""
    infer_pipe = pipe(pipeline, model_names=model)
    results = infer_pipe.run(request_body.dict())
    return dict(results=results)


### Model registry ###

@app.get("/registry/models", tags=["registry"])
async def list_registry_models():
    """List models available in the model registry."""
    return dict(results=tracking.list_models())


@app.get("/registry/models/{model_name}", tags=["registry"])
async def list_registry_model_versions(model_name: str):
    """List model versions available in the model registry."""
    return dict(results=tracking.list_model_versions(model_name))


@app.post("/registry/models/{model_name}/{version}", tags=["registry"])
async def deploy_model_version_from_registry(model_name: str, version: int):
    try:
        tracking.deploy_model_version(model_name, version)
        return dict(results="ok")
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.delete("/registry/models/{model_name}/{version}", tags=["registry"])
async def delete_model_version(model_name: str, version: int):
    try:
        tracking.delete_model_version(model_name, version)
        return dict(results="ok")
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
