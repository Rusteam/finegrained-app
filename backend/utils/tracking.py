import os
from urllib.parse import urljoin

from mlflow import MlflowClient
from pydantic import BaseModel

mlflow_client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))


class MLflowModel(BaseModel):
    """Model metadata."""

    name: str
    description: str | None
    last_updated_timestamp: int
    link: str


class MLflowModelVersion(BaseModel):
    """Model version metadata."""

    version: int
    last_updated_timestamp: int
    current_stage: str | None
    link: str


def _get_base_tracking_url(tracking_uri: str) -> str:
    """Get base tracking URL."""
    return tracking_uri.rstrip("#").rstrip("/")


def list_models() -> list[MLflowModel]:
    """List available models."""
    global mlflow_client
    res = mlflow_client.search_registered_models()
    base_url = _get_base_tracking_url(mlflow_client.tracking_uri)
    return [MLflowModel(
        **dict(model) | {"link": urljoin(base_url, f"/#/models/{model.name}")}
    ) for model in res]


def list_model_versions(model_name: str) -> list[MLflowModelVersion]:
    """List available models."""
    global mlflow_client
    model = mlflow_client.get_registered_model(model_name)
    base_url = _get_base_tracking_url(mlflow_client.tracking_uri)
    return [MLflowModelVersion(
        **dict(version) | {"link": urljoin(base_url, f"/#/models/{model.name}/versions/{version.version}")}
    ) for version in model.latest_versions]
