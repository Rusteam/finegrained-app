"""Connect to triton server nad perform inference
"""
import os

import numpy as np
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
from typing import List

from tritonclient.grpc import InferenceServerClient, InferInput, \
    InferRequestedOutput


def calc_num_batches(data_size: int, batch_size: int) -> int:
    n_batches, v = divmod(data_size, batch_size)
    if v > 0:
        n_batches += 1
    return n_batches


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask[..., np.newaxis].astype(np.float32)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    return np.sum(token_embeddings * input_mask_expanded,
                     1) / np.maximum(input_mask_expanded.sum(1), 1e-9)


class TritonClient(InferenceServerClient):
    def __init__(self, host=os.getenv("TRITON_HOST", "localhost:8001")):
        super().__init__(host)

    def list_models(self):
        repo_index = self.get_model_repository_index()
        models = [{
            "name": m.name,
            "version": m.version,
            "status": m.state
        } for m in repo_index.models
            if m.name.endswith("model")]
        return models

    def get_model_details(self, model_name: str) -> dict:
        meta = self.get_model_metadata(model_name)
        return dict(
            name=meta.name,
            inputs=self._parse_list(meta.inputs),
            outputs=self._parse_list(meta.outputs),
        )

    def _parse_inputs(self, input_config, raw_input) -> List[InferInput]:
        model_inputs = []
        for inp_conf in input_config:
            val = raw_input[inp_conf["name"].lower()]
            if not isinstance(val, (list, tuple)):
                val = [val]

            val_ar = np.array(val).astype(triton_to_np_dtype(inp_conf["datatype"]))
            if val_ar.ndim == 1 and len(inp_conf["shape"]) == 2:
                val_ar = val_ar[..., np.newaxis]

            inp = InferInput(inp_conf["name"], val_ar.shape,
                             inp_conf["datatype"])
            inp.set_data_from_numpy(val_ar)

            model_inputs.append(inp)
        return model_inputs

    def _get_batch_size(self, model_name) -> int:
        conf = self.get_model_config(model_name)
        return conf.config.max_batch_size

    def predict(self, model_name: str, raw_input, to_list=True):
        meta = self.get_model_details(model_name)
        model_outputs = self._parse_outputs(meta["outputs"])

        model_inputs = self._parse_inputs(meta["inputs"],
                                          raw_input,
                                          )

        resp = self.infer(model_name, model_inputs,
                        outputs=model_outputs)

        res = {out.name().lower(): resp.as_numpy(out.name()) for out in model_outputs}
        if "embeddings" in res and "attention_mask" in res:
            res["embeddings"] = mean_pooling(res["embeddings"],
                                             res["attention_mask"])
        if to_list:
            res = {k: v.tolist() for k, v in res.items()}
        return res

    def _parse_outputs(self, output_config):
        # TODO handle class count
        outs = [InferRequestedOutput(out["name"]) for out in output_config]
        return outs

    @staticmethod
    def _parse_list(meta):
        return [{
            "name": inp.name,
            "shape": inp.shape,
            "datatype": inp.datatype
        } for inp in meta]

