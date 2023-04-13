"""Connect to triton server and perform inference
"""
import os
from typing import List

import numpy as np
from tritonclient.grpc import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import triton_to_np_dtype


def calc_num_batches(data_size: int, batch_size: int) -> int:
    n_batches, v = divmod(data_size, batch_size)
    if v > 0:
        n_batches += 1
    return n_batches


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask[..., np.newaxis].astype(np.float32)
    input_mask_expanded = np.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    )
    return np.sum(token_embeddings * input_mask_expanded, 1) / np.maximum(
        input_mask_expanded.sum(1), 1e-9
    )


def _parse_one_class(class_prob: str) -> dict:
    prob, index, label = class_prob.decode().split(':')
    return dict(confidence=float(prob), index=int(index), label=label)


def parse_classes(class_probs: List[List[str]]) -> List[List[dict]]:
    """Convert triton format class predictions to list of dicts.

    Triton format: '0.999:1:label'

    Args:
        class_probs: a batch of top k predictions

    Returns:
        a batch (list) of top-k (list) class predictions (dict)
    """
    classes = [
        [_parse_one_class(one) for one in single_prediction]
        for single_prediction in class_probs
    ]
    return classes


class TritonClient(InferenceServerClient):
    def __init__(self, host=os.getenv("TRITON_HOST", "localhost:8001")):
        super().__init__(host)

    def list_models(self):
        repo_index = self.get_model_repository_index()
        models = [
            {"name": m.name, "version": m.version, "status": m.state}
            for m in repo_index.models
            if m.name.endswith("model")
        ]
        return models

    def get_model_details(self, model_name: str) -> dict:
        meta = self.get_model_metadata(model_name)
        return dict(
            name=meta.name,
            inputs=self._parse_list(meta.inputs),
            outputs=self._parse_list(meta.outputs),
        )

    def _parse_inputs(
        self, input_config, raw_input, batch_index
    ) -> List[InferInput]:
        start, end, squeeze = batch_index

        model_inputs = []
        for inp_conf in input_config:
            val = raw_input[inp_conf["name"].lower()][start:end]
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            if squeeze:
                val = val.squeeze(0)

            data_type = inp_conf["datatype"]
            val_ar = val.astype(triton_to_np_dtype(data_type))

            inp = InferInput(inp_conf["name"], val_ar.shape, data_type)
            inp.set_data_from_numpy(val_ar)

            model_inputs.append(inp)
        return model_inputs

    def _parse_outputs(self, output_config, top_k=0):
        outs = [
            InferRequestedOutput(out["name"], class_count=top_k)
            for out in output_config
        ]
        return outs

    @staticmethod
    def _parse_list(meta):
        return [
            {"name": inp.name, "shape": list(inp.shape), "datatype": inp.datatype}
            for inp in meta
        ]

    def _get_batch_size(self, model_name) -> int:
        conf = self.get_model_config(model_name)
        return conf.config.max_batch_size

    def predict(self, model_name: str, raw_input, to_list=True, **kwargs):
        meta = self.get_model_details(model_name)
        model_outputs = self._parse_outputs(meta["outputs"], **kwargs)
        batch_size = self._get_batch_size(model_name)
        input_len = max(len(v) for v in raw_input.values() if v is not None)

        if batch_size == 0:
            batch_size += 1
            squeeze = True
        else:
            squeeze = False

        outputs = [
            self._predict_batch(
                model_name,
                raw_input,
                meta["inputs"],
                model_outputs,
                batch=(i, i + batch_size, squeeze),
            )
            for i in range(0, input_len, batch_size)
        ]

        for one in model_outputs:
            # TODO remove this clause once fixed
            if one.name().lower() == "attention_mask":
                model_outputs.remove(one)
        res = {
            model_out.name().lower(): np.vstack([
                out[model_out.name().lower()] for out in outputs
            ])
            for model_out in model_outputs
        }

        if to_list:
            res = {k: v.tolist() for k, v in res.items()}

        if kwargs.get("top_k", 0) > 0:
            res = {k: parse_classes(v) for k, v in res.items()}

        return res

    def _predict_batch(
        self, model_name, raw_input, input_config, model_outputs, batch
    ):
        model_inputs = self._parse_inputs(
            input_config,
            raw_input,
            batch
        )

        resp = self.infer(model_name, model_inputs, outputs=model_outputs)

        res = {
            out.name().lower(): resp.as_numpy(out.name())
            for out in model_outputs
        }

        # TODO to integrate into a model
        if "embeddings" in res and "attention_mask" in res:
            res["embeddings"] = mean_pooling(
                res["embeddings"], res.pop("attention_mask")
            )

        return res
