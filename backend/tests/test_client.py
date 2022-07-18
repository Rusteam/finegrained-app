import random

import os
from typing import List
from pathlib import Path

import numpy as np

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app.routes import app
from backend.utils.image_utils import encode_image_base64, load_base64_image


def dummy_image_base64(x):
    img = np.random.randint(0, 255, size=(x + 19, x - 19, 3), dtype=np.uint8)
    b64 = encode_image_base64(Image.fromarray(img))
    return b64


def real_base64_image():
    path = Path(__file__).parent / "images/img_3b.jpg"
    b64 = load_base64_image(str(path))
    return b64


MODELS = [
    dict(
        name="sentence_similarity_model",
        # request: (request body, request params, expected output shape)
        requests=[
            (
                {
                    "text": [  # many texts with differing lengths
                        "Скажите, а сколько у меня пользовтелей в этом боте "
                        "щас? я просто хочу понять насколько мне нужно "
                        "переходить на платную версию",
                        "Где посмотреть список пользователей бота?",
                        "Как узнать количество людей, которые открыли, "
                        "стартовали бота",
                        "список пользователь\nпользователей\nСсылка на "
                        "список подписчиков (URL)",
                        "Здравствуйте! А где сменить пароль?",
                        "Как поменять пароль от личного кабинета?",
                        "Можно ли изменить пароль личного кабинета",
                        "здравствуйте можно ли как то с одного квеста "
                        "перенести часть компонентов в другой, при этом не "
                        "создавая заново все?",
                    ]
                },
                {},
                dict(embeddings=(8, 768)),
            ),
            (
                {"text": "Just a single text sentence"},
                {},
                dict(embeddings=(1, 768)),
            ),
        ],
    ),
    dict(
        name="image_classification_model",
        requests=[
            (
                {"image": [dummy_image_base64(270), dummy_image_base64(301)]},
                dict(top_k=5),
                dict(class_probs=(2, 5)),
            ),
            (
                {"image": dummy_image_base64(280)},
                dict(top_k=3),
                dict(class_probs=(1, 3)),
            ),
        ],
    ),
    dict(
        name="object_detection_model",
        requests=[
            (
                {"image": real_base64_image()},
                {},
                dict(boxes=(8, 4), scores=(8, 1), class_probs=(8, 1)),
            ),
        ],
    ),
]
PIPELINES = [
    dict(
        name="object-recognition",
        config=dict(model_names=["detector", "classifier"]),
        requests=[
            (
                {"image": real_base64_image()},
                dict(
                    model=[
                        "object_detection_model",
                        "image_classification_model",
                    ]
                ),
                dict(n=8),
            )
        ],
    )
]


def _get_names(values: List[dict]) -> List[str]:
    return [v["name"] for v in values]


def _get_requests(values: List[dict]):
    return [[m["name"]] + list(r) for m in values for r in m["requests"]]


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

    models = resp.json()["results"]
    assert isinstance(models, list)
    assert len(models) == len(MODELS)
    assert all([m["status"] == "READY" for m in models])
    actual_names = [m["name"] for m in models]
    for exp in MODELS:
        assert exp["name"] in actual_names


@pytest.mark.parametrize("model_name", _get_names(MODELS))
def test_model_details(client, model_name):
    resp = client.get(f"/models/{model_name}")
    assert resp.status_code == 200, resp.content.decode()


@pytest.mark.parametrize(
    "model_name,request_body,params,expected_output",
    _get_requests(MODELS),
)
def test_models_predict(
    client, model_name, request_body, params, expected_output
):
    resp = client.post(
        f"/models/{model_name}/predict", json=request_body, params=params
    )
    assert resp.status_code == 200, resp.content.decode()

    results = resp.json()["results"]
    for k, v in expected_output.items():
        assert np.array(results[k]).shape == v


@pytest.mark.parametrize(
    "data_name,request_body,expected_output",
    [
        (
            "test_data",
            {
                "vectors": [
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
def test_add_vectors(
    client, data_name, request_body, expected_output, tmp_path
):
    os.environ["VECTOR_STORAGE"] = str(tmp_path)
    resp = client.post(f"/embeddings/{data_name}", json=request_body)
    assert resp.status_code == 200, resp.content
    assert resp.json()["results"] == expected_output

    top_k = 2
    search_resp = client.post(
        f"/embeddings/{data_name}/search",
        json=request_body | dict(top_k=top_k),
    )
    assert search_resp.status_code == 200, search_resp.content
    for one in search_resp.json()["results"]:
        assert len(one) == top_k


@pytest.fixture(scope="module")
def vector_db(client):
    db_name = "test_db"
    db = [
        "Скажите, а сколько у меня пользовтелей в этом боте щас? я просто "
        "хочу понять насколько мне нужно переходить на платную версию",
        "Где посмотреть список пользователей бота?",
        "Как узнать количество людей, которые открыли, стартовали бота",
        "список пользователь\nпользователей\nСсылка на список подписчиков ("
        "URL)",
        "Здравствуйте! А где сменить пароль?",
        "Как поменять пароль от личного кабинета?",
        "Можно ли изменить пароль личного кабинета",
        "здравствуйте можно ли как то с одного квеста перенести часть "
        "компонентов в другой, при этом не создавая заново все?",
    ]
    resp = client.post(
        "/models/sentence_similarity_model/predict", json={"text": db}
    ).json()["results"]
    _ = client.post(
        f"/embeddings/{db_name}",
        json={
            "vectors": resp["embeddings"],
            "data": [
                dict(
                    index=i,
                    url="http://url.com",
                    other=None,
                    group=random.choice(["A", "B"]),
                )
                for i in range(len(db))
            ],
        },
    )
    return db_name


@pytest.mark.parametrize(
    "groupby,top_k", [(None, 3), (None, 1), ("group", 2), ("group", 1)]
)
def test_search_similar(client, vector_db, groupby, top_k):
    n_q = 2
    q_vectors = np.random.randn(n_q, 768).tolist()
    resp = client.post(
        f"/embeddings/{vector_db}/search",
        json=dict(vectors=q_vectors, top_k=top_k, groupby=groupby),
    )
    assert resp.status_code == 200

    results = resp.json()["results"]

    _check_search_results(results, n_expected=n_q, top_k=top_k, groupby=groupby)


@pytest.mark.parametrize(
    "model_name,text,top_k,squeeze,groupby",
    [
        ("sentence_similarity_model", "Simple query request", 2, False, None),
        ("sentence_similarity_model", "Simple query request", 1, True, "group"),
        ("sentence_similarity_model", ["List of", "short queries"], 2, False, "group"),
        ("sentence_similarity_model", ["List of", "short queries"], 4, True, None),
    ],
)
def test_models_predict_and_search(
    client, vector_db, model_name, text, top_k, squeeze, groupby
):
    resp = client.post(
        f"/models/{model_name}/search/{vector_db}",
        params=dict(top_k=top_k, squeeze=squeeze, groupby=groupby),
        json={"text": text},
    )
    assert resp.status_code == 200, resp.content.decode()

    out = resp.json()["results"]
    n_request = len(text) if isinstance(text, (list, tuple)) else 1
    assert n_request == len(out)
    assert isinstance(out, list)

    if squeeze and n_request == 1:
        out = [out]

    _check_search_results(out, n_expected=n_request, top_k=top_k, groupby=groupby)


def test_list_pipelines(client):
    resp = client.get("/pipelines")
    assert resp.status_code == 200

    results = resp.json()["results"]
    assert len(results) == len(PIPELINES)
    for p in PIPELINES:
        assert p["name"] in results


@pytest.mark.parametrize(
    "pipe_name,expected", [(p["name"], p["config"]) for p in PIPELINES]
)
def test_get_pipeline_config(client, pipe_name, expected):
    resp = client.get(f"/pipelines/{pipe_name}")
    assert resp.status_code == 200

    results = resp.json()["results"]
    assert results == expected


@pytest.mark.parametrize(
    "pipe_name,request_body,params,expected", _get_requests(PIPELINES)
)
def test_run_pipeline(client, pipe_name, request_body, params, expected):
    resp = client.post(
        f"/pipelines/{pipe_name}/predict", json=request_body, params=params
    )
    assert resp.status_code == 200, resp.content.decode()

    results = resp.json()["results"]
    assert len(results) == expected["n"]
    assert all([isinstance(one, dict) for one in results])


def _check_search_results(results, n_expected, top_k, groupby):
    assert len(results) == n_expected

    for one in results:
        assert len(one) == top_k

        assert all(["index" in o for o in one])

        if groupby:
            key_vals = [o[groupby] for o in one]
            assert len(key_vals) == len(set(key_vals))
