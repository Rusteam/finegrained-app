import shutil
from pathlib import Path

import mlflow.onnx


class ModelRepository:
    """Manage new model deployments from MLflow to Triton.
    """
    def __init__(self, triton_repository: str):
        self.triton_repository = Path(triton_repository).resolve()

    def _create_version_dir(self, model_name: str, version: int) -> Path:
        version_dir = self._get_model_version_dir(model_name, version)
        if version_dir.exists():
            raise FileExistsError(f"{model_name=!r} with {version=} already exists")
        version_dir.mkdir(parents=True)
        return version_dir

    def create_model_version(self, model_name: str, version: int, model_uri: str):
        """Deploy a new model version.

        Args:
            model_name: name of the model
            version: version number
            model_uri: mlflow model uri
        """
        version_dir = self._create_version_dir(model_name, version)
        mlflow.onnx.load_model(model_uri, dst_path=str(version_dir))

    def delete_model_version(self, model_name: str, version: int):
        """Delete a model version.

        Args:
            model_name: name of the model
            version: version number
        """
        version_dir = self._get_model_version_dir(model_name, version)
        if not version_dir.exists():
            raise FileNotFoundError(f"{model_name=!r} with {version=} does not exist")
        shutil.rmtree(version_dir)

    def _get_model_version_dir(self, model_name: str, version: int) -> Path:
        return self.triton_repository / model_name / str(version)
