"""Create model pipelines.
"""
from typing import List, Dict, Optional

import numpy as np
from dataclasses import dataclass

from .triton import TritonClient


@dataclass
class PipelineMixin:
    """Base class that other pipelines must inherit from.
    """
    triton: TritonClient

    def not_implemented(self):
        raise NotImplementedError("Subclass has to implement this method")

    @property
    def model_names(self) -> List[str]:
        """Define getter/setter for each model.
        """
        return []

    def get_config(self):
        self.not_implemented()

    def run(self):
        """This method is called when pipeline runs.
        """
        self.not_implemented()

    def set_models(self, models):
        if len(models) != (n_required := len(self.model_names)):
            raise ValueError(f"{n_required} models are required.")
        for key, value in zip(self.model_names, models):
            setattr(self, key, value)


class ObjectRecognition(PipelineMixin):
    """Object recognition sends a request to object detector first,
    then crops results and sends them to image classification.
    """
    @property
    def model_names(self):
        return ["detector", "classifier"]

    @property
    def detector(self) -> str:
        return self._detector

    @detector.setter
    def detector(self, value):
        self._detector = value

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    def get_config(self):
        return {"model_names": self.model_names}

    def _make_box(
        self, src_image: np.ndarray, bounds: List[float]
    ) -> np.ndarray:
        x, y, w, h = bounds
        img_h, img_w = src_image.shape[:2]

        xmin = max(0, int((x - w * 0.5) * img_w))
        xmax = min(img_w, int((x + w * 0.5) * img_w))
        ymin = max(0, int((y - h * 0.5) * img_h))
        ymax = min(img_h, int((y + h * 0.5) * img_h))

        box = src_image[ymin:ymax, xmin:xmax]
        return box

    def _crop_boxes(
        self, image: np.ndarray, detections: Dict[str, list]
    ) -> List[np.ndarray]:
        boxes = [
            self._make_box(image, bounds) for bounds in detections["boxes"]
        ]
        return dict(image=boxes)

    def _merge(
        self, detections: Dict[str, list], class_predictions
    ) -> List[dict]:
        merged = [
            dict(box=box, box_score=box_score[0]) | clf[0]
            for box, box_score, clf in zip(
                detections["boxes"],
                detections["scores"],
                class_predictions["class_probs"],
            )
        ]
        return merged

    def run(self, raw_input):
        detections = self.triton.predict(self.detector, raw_input)
        crops = self._crop_boxes(raw_input["image"][0], detections)
        class_predictions = self.triton.predict(
            self.classifier, crops, top_k=1
        )
        merged = self._merge(detections, class_predictions)
        return merged


@dataclass
class Pipelines:
    """This class handles pipeline listing and initialization.

    Usage:
        pipe = Pipelines(triton_client)
        print(pipe.list())
        infer_pipe = pipe(pipeline_name, model_names=[list, of, models])
        results = infer_pipe.run(raw_input)
    """
    triton: TritonClient
    pipelines = {"object-recognition": ObjectRecognition}

    def list(self):
        return list(self.pipelines)

    def __call__(self, name: str, model_names: Optional[List[str]] = None):
        if name in self.pipelines:
            pipe = self.pipelines[name](self.triton)
            if model_names:
                pipe.set_models(model_names)
            return pipe
        else:
            raise ValueError(f"pipeline {name=} not found!")
