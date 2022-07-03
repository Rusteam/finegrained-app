"""Index images and search
"""
import json

from dataclasses import dataclass

from pathlib import Path
from typing import List, Union, Tuple

import faiss
import numpy as np

from backend.utils.triton import TritonClient


INPUT_TYPE = Union[str, np.ndarray]
TOP_SIM_TYPE = List[List[Tuple[str, float]]]


def write_lines(data: List[str], write_path: Path) -> None:
    write_path.write_text("\n".join(data))
    print(f"{len(data)} rows saved to {str(write_path)}")


def read_lines(src_file: Path) -> List[str]:
    data = src_file.read_text().strip().split("\n")
    return data


def write_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f)


def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def to_float(sample: np.ndarray) -> List[np.ndarray]:
    sample_float = (sample / 255.0).astype(np.float32)
    return sample_float


@dataclass
class FeatureExtractor:
    model: TritonClient

    def _run_one(self, sample: np.ndarray) -> np.ndarray:
        out, *_ = self.model.predict([to_float(sample)])
        return out

    def __call__(self, samples: List[np.ndarray]) -> np.ndarray:
        outputs = [self._run_one(smp) for smp in samples]
        outputs = np.vstack(outputs)
        return outputs


def init_feature_extractor(model_name: str) -> FeatureExtractor:
    triton_client = TritonClient(model_name)
    model = FeatureExtractor(model=triton_client)
    return model


class SimilaritySearch:
    def __init__(self):
        pass

    def from_file(self, filepath):
        index = faiss.read_index(filepath)
        data = read_json(self._make_data_path(filepath))
        return index, data

    @staticmethod
    def _make_data_path(index_file: str) -> Path:
        dest = Path(index_file)
        labels_file = dest.parent / f"{dest.stem}.json"
        return labels_file

    def _prepare_output(self, I: np.ndarray, D: np.ndarray, data):
        results = [
            [data[i] | {"similarity": float(p)} for i, p in zip(labels, probs)]
            for labels, probs in zip(I, D)
        ]
        return results

    def _prepare_vectors(self, vectors: Union[List, np.ndarray]):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def index(self, vectors: Union[List, np.ndarray], dest: str, data: List[dict]):
        vectors = self._prepare_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, dest)
        write_json(data, self._make_data_path(dest))
        return {"index": index.ntotal}

    def query_vectors(self, index_file, vectors: np.ndarray, top_k: int = 1):
        index, data = self.from_file(index_file)
        vectors = self._prepare_vectors(vectors)
        D, I = index.search(vectors, top_k)
        results = self._prepare_output(I, D, data)
        return results
