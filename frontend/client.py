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
        headers = kwargs.pop("headers", {}) | {"content-type": "application/json"}
        data = json.dumps(kwargs.pop("data", {}))
        request = requests.request(method, url, headers=headers, data=data,
                                    timeout=60 if bool(data) else 5,
                                   **kwargs)
        request.raise_for_status()
        return request.json()

    def list_models(self):
        models = self._make_request("get", "/models")
        return [m["name"] for m in models]

    def predict(self, model_name: str, data):
        out = self._make_request("post", f"/models/{model_name}/predict",
                                 data=data)
        return out

    def index(self, data_name: str, vectors, data):
        resp = self._make_request("post", f"/embeddings/{data_name}",
                                  data=dict(
                                      features=vectors,
                                      data=data,
                                  ))
        return resp

    def search_similar(self, data_name: str, vectors, top_k=3):
        resp = self._make_request("post", f"/embeddings/{data_name}/search",
                                  data=dict(features=vectors, top_k=top_k))
        return resp
