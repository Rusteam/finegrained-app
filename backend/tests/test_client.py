import os

import numpy as np

import pytest
from fastapi.testclient import TestClient

from backend.app.routes import app


@pytest.fixture(scope="module")
def client():
    client = TestClient(app)
    return client


def test_hello(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Hello World"}


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200

    parsed = resp.json()
    assert isinstance(parsed, list)
    assert len(parsed) > 0
    assert isinstance(parsed[0], dict)


@pytest.mark.parametrize("model_name", ["sentence_similarity_model"])
def test_model_details(client, model_name):
    resp = client.get(f"/models/{model_name}")
    assert resp.status_code == 200, resp.content


@pytest.mark.parametrize(
    "model_name,request_body,expected_output",
    [
        (
            "sentence_similarity_model",
            {"text": ["Sentence one", "and another sentence",
                      "more coming now", "and a last one"]},
            dict(embeddings=(2, 5, 768), attention_mask=(2, 5)),
        )
    ],
)
def test_models_predict(client, model_name, request_body, expected_output):
    resp = client.post(f"/models/{model_name}/predict", json=request_body)
    assert resp.status_code == 200, resp.content.decode()

    out = resp.json()
    for k, v in expected_output.items():
        assert np.array(out[k]).shape == v


@pytest.mark.parametrize(
    "data_name,request_body,expected_output",
    [
        (
            "test_data",
            {
                "features": [
                    [1.0, 0.5, 1.8],
                    [0.2, 0.5, 0.66],
                    [0.01, -0.33, 0.23],
                ],
                "data": [{"msg": 1}, {"msg": 2}, {"msg": 3}],
            },
            dict(index=3),
        )
    ],
)
def test_add_vectors(client, data_name, request_body, expected_output,
                     tmp_path):
    os.setenv("VECTOR_STORAGE", tmp_path)
    resp = client.post(f"/embeddings/{data_name}", json=request_body)
    assert resp.status_code == 200, resp.content
    assert resp.json() == expected_output

    top_k = 2
    search_resp = client.post(f"/embeddings/{data_name}/search",
                              json=request_body | dict(top_k=top_k))
    assert search_resp.status_code == 200, search_resp.content
    for one in search_resp.json():
        assert len(one) == top_k

