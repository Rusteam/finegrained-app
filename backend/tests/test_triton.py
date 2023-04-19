import numpy as np
import pytest
from tritonclient.grpc import InferResult

from utils.triton import TritonClient


@pytest.fixture(scope="module")
def triton_client():
    client = TritonClient()
    return client


@pytest.fixture(scope="module")
def raw_image():
    return dict(
        image=[np.random.rand(64, 64, 3).astype(np.float32)]
    )


@pytest.fixture(scope="module")
def embedding_size():
    return 32


@pytest.fixture()
def triton_mock(triton_client, mocker, embedding_size):
    triton_client.get_model_details = mocker.Mock(return_value={'name': 'feature_extractor_model', 'inputs': [{'name': 'IMAGE', 'shape': [-1, -1, -1, 3], 'datatype': 'UINT8'}], 'outputs': [{'name': 'FEATURES', 'shape': [-1, embedding_size], 'datatype': 'FP32'}]})
    triton_client._get_batch_size = mocker.Mock(return_value=1)
    triton_client.is_model_ready = mocker.Mock(return_value=True)
    InferResult.as_numpy = mocker.Mock(return_value=np.random.rand(1, embedding_size).astype(np.float32))
    triton_client.infer = mocker.Mock(return_value=InferResult)
    triton_client.load_model = mocker.Mock()
    return mocker


def test_feature_extractor(triton_client, raw_image, triton_mock, embedding_size):
    prediction = triton_client.predict("feature_extractor_model",
                                       raw_input=raw_image)

    assert len(prediction) == 1
    assert "features" in prediction and len(prediction["features"]) == 1
    assert len(prediction["features"][0]) == embedding_size
