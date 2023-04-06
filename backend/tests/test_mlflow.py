import pytest
from mlflow.entities.model_registry import RegisteredModel, ModelVersion

from ..utils import tracking


@pytest.fixture
def mlflow_client_mock(mocker):
    mock = mocker.patch("backend.utils.tracking.mlflow_client")
    mock.tracking_uri = "http://localhost:5000/#"
    return mock


@pytest.fixture
def mlflow_models_mock(mlflow_client_mock):
    mlflow_client_mock.search_registered_models.return_value = [
        RegisteredModel(name="identity_matrix", creation_timestamp=0, last_updated_timestamp=3),
        RegisteredModel(name="matrix_two", creation_timestamp=1, last_updated_timestamp=2),
    ]
    return mlflow_client_mock


@pytest.fixture
def mlflow_model_versions_mock(mlflow_client_mock):
    mlflow_client_mock.get_registered_model.return_value = RegisteredModel(
        name="identity_matrix", creation_timestamp=0, last_updated_timestamp=3,
        latest_versions=[
            ModelVersion(name="identity_matrix", version="1", creation_timestamp=0,
                         last_updated_timestamp=0, current_stage="Staging", run_id="1"),
            ModelVersion(name="identity_matrix", version="2", creation_timestamp=1,
                         last_updated_timestamp=2, current_stage="Production", run_id="2"),
            ModelVersion(name="identity_matrix", version="3", creation_timestamp=3,
                         last_updated_timestamp=3, current_stage=None, run_id="3"),
        ]
    )
    return mlflow_client_mock


def test_list_models(mlflow_models_mock):
    models = tracking.list_models()
    assert len(models) == 2
    assert models[0].name == "identity_matrix"
    assert models[0].link == "http://localhost:5000/#/models/identity_matrix"
    assert models[1].name == "matrix_two"
    assert models[1].link == "http://localhost:5000/#/models/matrix_two"
    mlflow_models_mock.search_registered_models.assert_called_once()


def test_list_model_versions(mlflow_model_versions_mock):
    # test model versions
    versions = tracking.list_model_versions("identity_matrix")
    assert len(versions) == 3
    assert versions[0].version == 1
    assert versions[1].current_stage == "Production"
    assert versions[2].link == "http://localhost:5000/#/models/identity_matrix/versions/3"
