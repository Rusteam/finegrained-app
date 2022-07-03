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
            {
                "text": [
                    "Скажите, а сколько у меня пользовтелей в этом боте "
                    "щас? я просто хочу понять насколько мне нужно "
                    "переходить на платную версию",
                    "Где посмотреть список пользователей бота?",
                    "Как узнать количество людей, которые открыли, " "стартовали бота",
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
            dict(embeddings=(8, 768), attention_mask=(8, 34)),
        ),
        (
            "sentence_similarity_model",
            {"text": "Just a single text sentence"},
            dict(embeddings=(1, 768), attention_mask=(1, 7)),
        ),
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
def test_add_vectors(client, data_name, request_body, expected_output, tmp_path):
    os.setenv("VECTOR_STORAGE", tmp_path)
    resp = client.post(f"/embeddings/{data_name}", json=request_body)
    assert resp.status_code == 200, resp.content
    assert resp.json() == expected_output

    top_k = 2
    search_resp = client.post(
        f"/embeddings/{data_name}/search",
        json=request_body | dict(top_k=top_k),
    )
    assert search_resp.status_code == 200, search_resp.content
    for one in search_resp.json():
        assert len(one) == top_k


@pytest.fixture(scope="module")
def vector_db(client):
    db_name = "test_db"
    db = [
        "Скажите, а сколько у меня пользовтелей в этом боте щас? я просто "
        "хочу понять насколько мне нужно переходить на платную версию",
        "Где посмотреть список пользователей бота?",
        "Как узнать количество людей, которые открыли, стартовали бота",
        "список пользователь\nпользователей\nСсылка на список подписчиков (" "URL)",
        "Здравствуйте! А где сменить пароль?",
        "Как поменять пароль от личного кабинета?",
        "Можно ли изменить пароль личного кабинета",
        "здравствуйте можно ли как то с одного квеста перенести часть "
        "компонентов в другой, при этом не создавая заново все?",
    ]
    resp = client.post("/models/sentence_similarity_model/predict", json={"text": db})
    _ = client.post(
        f"/embeddings/{db_name}",
        json={
            "vectors": resp.json()["embeddings"],
            "data": [dict(index=i) for i in range(len(db))],
        },
    )
    return db_name


@pytest.mark.parametrize(
    "model_name,text,top_k,squeeze",
    [
        ("sentence_similarity_model", "Simple query request", 2, False),
        ("sentence_similarity_model", "Simple query request", 1, True),
        ("sentence_similarity_model", ["List of", "short queries"], 4, False),
        ("sentence_similarity_model", ["List of", "short queries"], 4, True),
    ],
)
def test_models_predict_and_search(client, vector_db, model_name, text, top_k, squeeze):
    resp = client.post(
        f"/models/{model_name}/search/{vector_db}",
        params=dict(top_k=top_k, squeeze=squeeze),
        json={"text": text},
    )
    assert resp.status_code == 200, resp.content.decode()

    out = resp.json()["results"]
    n_request = len(text) if isinstance(text, (list, tuple)) else 1
    assert n_request == len(out)
    assert isinstance(out, list)

    if squeeze and n_request == 1:
        out = [out]

    for item in out:
        assert isinstance(item, list)
        assert len(item) == top_k
        for top in item:
            assert isinstance(top, dict)
            assert "index" in top
