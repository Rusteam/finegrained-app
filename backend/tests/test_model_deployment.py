import pytest

from ..utils.model_deployment import ModelRepository


@pytest.fixture(scope="module")
def model_repository(tmp_path_factory):
    return ModelRepository(str(tmp_path_factory.mktemp("triton")))


@pytest.fixture(scope="module")
def model_uri():
    return "models:/finegrained/1"


@pytest.fixture
def mlflow_mock(mocker):
    mock = mocker.patch("backend.utils.model_deployment.mlflow.onnx.load_model")
    return mock


@pytest.mark.parametrize("model_name,version", [
    ("softmax", 1),
    ("softmax", 2),
    ("other", 3),
])
def test_create_model_version(model_repository, model_name, version, model_uri, mlflow_mock):
    model_repository.create_model_version(model_name, version, model_uri)

    expected_path = model_repository.triton_repository / model_name / str(version)
    mlflow_mock.assert_called_with(model_uri, dst_path=str(expected_path))
    assert expected_path.exists()


def test_delete_model_version(model_repository, model_uri, mlflow_mock):
    model_name = "model"
    version = 2
    model_repository.create_model_version(model_name, version, model_uri)
    model_repository.delete_model_version(model_name, version)

    expected_path = model_repository.triton_repository / model_name / str(version)
    assert not expected_path.exists()
