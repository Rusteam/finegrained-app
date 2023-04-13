import os
from urllib.parse import urljoin, urlparse

from mlflow import MlflowClient
from pydantic import BaseModel

from .model_deployment import ModelRepository

mlflow_client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))
model_repository = ModelRepository(os.getenv("TRITON_MODEL_REPOSITORY",
                                             "./models/triton"))


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
    """Get base tracking URL removing user and password from URI."""
    parsed = urlparse(tracking_uri)
    if parsed.username and parsed.password:
        netloc = parsed.netloc.split("@", 1)[1]
    else:
        netloc = parsed.netloc
    return f"{parsed.scheme}://{netloc}"


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


def deploy_model_version(model_name: str, version: int):
    """Deploy model version to triton."""
    global mlflow_client, model_repository
    model_uri = f"models:/{model_name}/{version}"
    model_repository.create_model_version(model_name, version, model_uri)


def delete_model_version(model_name: str, version: int):
    """Delete model version from triton."""
    global model_repository
    model_repository.delete_model_version(model_name, version)
