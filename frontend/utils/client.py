"""A client to communicate with backend server.
"""
import json

from urllib.parse import urljoin

import requests
from dataclasses import dataclass

import os


@dataclass
class Client:
    url = os.getenv("BACKEND_URL", "http://localhost:8100")

    def _make_request(self, method: str, path: str, **kwargs):
        url = urljoin(self.url, path)
        headers = kwargs.pop("headers", {}) | {
            "content-type": "application/json"
        }
        data = json.dumps(kwargs.pop("data", {}))
        request = requests.request(
            method,
            url,
            headers=headers,
            data=data,
            timeout=60 if bool(data) else 5,
            **kwargs,
        )
        if request.status_code != 200:
            raise requests.exceptions.HTTPError(request.content.decode())
        return request.json()["results"]

    def list_models(self):
        models = self._make_request("get", "/models")
        return [m["name"] for m in models]

    def list_embeddings(self):
        embeddings = self._make_request("get", "/embeddings")
        return embeddings

    def list_pipelines(self):
        pipe_names = self._make_request("get", "/pipelines")
        return pipe_names

    def index(self, data_name: str, vectors, data):
        resp = self._make_request(
            "post",
            f"/embeddings/{data_name}",
            data=dict(
                vectors=vectors,
                data=data,
            ),
        )
        return resp

    def predict(self, model_name: str, data, **kwargs):
        out = self._make_request(
            "post", f"/models/{model_name}/predict", data=data, **kwargs
        )
        return out

    def predict_and_search(
        self, model_name: str, data_name: str, data, **kwargs
    ):
        resp = self._make_request(
            "post",
            f"/models/{model_name}/search/{data_name}",
            data=data,
            **kwargs
        )
        return resp

    def run_pipeline(self, pipeline_name: str, data, **kwargs):
        out = self._make_request(
            "post", f"/pipelines/{pipeline_name}/predict", data=data, **kwargs
        )
        return out

    def search_similar(self, data_name: str, vectors, top_k=3):
        resp = self._make_request(
            "post",
            f"/embeddings/{data_name}/search",
            data=dict(vectors=vectors),
            params=dict(top_k=top_k),
        )
        return resp

    def list_registry_models(self) -> list[dict]:
        return self._make_request("get", "/registry/models")

    def list_registry_model_versions(self, model: str) -> list[dict]:
        return self._make_request("get", f"/registry/models/{model}")

    def deploy_model_version_from_registry(self, model: str, version: int):
        return self._make_request(
            "post", f"/registry/models/{model}/{version}"
        )

    def delete_model_version(self, model: str, version: int):
        return self._make_request(
            "delete", f"/registry/models/{model}/{version}"
        )
