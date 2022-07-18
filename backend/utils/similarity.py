"""Index images and search
"""
import itertools

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional

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


def _groupby(results: List[dict], field: str, limit: int) -> List[dict]:
    group_fn = lambda x: x[field]
    distinct = []
    for key, group in itertools.groupby(results, key=group_fn):
        group = list(group)
        sorted(group, key=lambda x: x["similarity"], reverse=True)
        distinct.append(group[0])
    return distinct[:limit]


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

    @staticmethod
    def _prepare_vectors(vectors: Union[List, np.ndarray]) -> np.ndarray:
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    @staticmethod
    def _prepare_output(
        I: np.ndarray,
        D: np.ndarray,
        data,
        groupby: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[List[dict]]:
        results = [
            [data[i] | {"similarity": float(p)} for i, p in zip(labels, probs)]
            for labels, probs in zip(I, D)
        ]
        if bool(groupby):
            assert limit is not None
            results = [_groupby(res, groupby, limit) for res in results]
        return results

    def index(
        self, vectors: Union[List, np.ndarray], dest: str, data: List[dict]
    ):
        """Create a new faiss index and save it.

        Vectors will be saved to dest file and data will be saved to
        (dest.stem).json file.

        Args:
            vectors: embeddings to build the index
            dest: where to save embeddings
            data: data will be saved along with vectors
        """
        vectors = self._prepare_vectors(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, dest)
        write_json(data, self._make_data_path(dest))
        return {"index": index.ntotal}

    def query_vectors(
        self,
        index_file,
        vectors: np.ndarray,
        top_k: int = 1,
        groupby: Optional[str] = None,
    ) -> List[List[Dict]]:
        """Find top matching samples and return data and vectors.

        Args:
            index_file: faiss index file
            vectors: query vectors (saved vectors are matched against these)
            top_k: number of top matches to return
            groupby: group results by this key and take top element only

        Returns:
            a list of top matches for each query vector.
        """
        index, data = self.from_file(index_file)
        vectors = self._prepare_vectors(vectors)
        D, I = index.search(vectors, top_k * 4 if groupby else top_k)
        results = self._prepare_output(
            I, D, data, groupby=groupby, limit=top_k
        )
        return results
