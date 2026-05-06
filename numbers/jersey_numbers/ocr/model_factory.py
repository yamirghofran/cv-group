"""Factory for creating OCR models by name.

Hardcoded checkpoint paths - edit CHECKPOINT_PATHS to change model locations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

from .resnet18 import ResNet18OCR
from .resnet34 import ResNetOCR
from .smolvlm2 import RoboflowOCR

# Hardcoded checkpoint paths for local models
CHECKPOINT_PATHS = {
    "resnet18": "numbers/models/resnet18.pth",
    "resnet34": "numbers/models/resnet34.pth",
}

# API model settings
API_MODELS = {
    "smol": "jersey-number-ocr/1",  # Update with your actual Roboflow model ID
    "smolvlm2": "jersey-number-ocr/1",
}


def create_ocr_model(
    model_name: Literal["resnet", "resnet18", "x2", "resnet34", "smol", "smolvlm2"],
    device: str = "auto",
):
    """
    Create an OCR model by name.

    Args:
        model_name: Model identifier - one of "resnet", "x2", "smol"
        device: Device to run inference on ("auto", "cpu", "cuda", "mps")

    Returns:
        Model instance with predict(bgr_image) -> (str, float) interface

    Raises:
        ValueError: If model_name is not recognized
        FileNotFoundError: If checkpoint file doesn't exist
    """
    name = model_name.lower().strip()

    # Handle API-based models (smol/smolvlm2)
    if name in API_MODELS:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"ROBOFLOW_API_KEY environment variable required for model '{model_name}'"
            )
        model_id = API_MODELS[name]
        return RoboflowOCR(model_id=model_id, api_key=api_key)

    # Handle local models
    if name not in CHECKPOINT_PATHS:
        all_models = list(CHECKPOINT_PATHS.keys()) + list(API_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {all_models}")

    checkpoint_path = Path(CHECKPOINT_PATHS[name])

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please ensure the model checkpoint exists or update CHECKPOINT_PATHS."
        )

    # Route to appropriate model class
    if name in ("resnet", "resnet18"):
        return ResNet18OCR(checkpoint_path, device_str=device)
    elif name in ("x2", "resnet34"):
        return ResNetOCR(checkpoint_path, device_str=device)
    else:
        raise ValueError(f"Model '{model_name}' not implemented")


class NumberOnlyModel:
    """Wrapper that returns only the predicted number, no confidence score."""

    def __init__(self, wrapped_model) -> None:
        self._model = wrapped_model

    def predict(self, bgr_image: np.ndarray) -> str:
        """Predict jersey number - returns only the number string."""
        number, _ = self._model.predict(bgr_image)
        return number

    def predict_batch(self, bgr_images: list[np.ndarray]) -> list[str]:
        """Predict on batch - returns list of number strings."""
        return [self.predict(img) for img in bgr_images]


def predict_number(model_name: str, bgr_image: np.ndarray, device: str = "auto") -> str:
    """
    One-shot prediction: create model and predict on a single image.

    Args:
        model_name: Model identifier
        bgr_image: BGR image crop (numpy array from bounding box)
        device: Device for inference

    Returns:
        predicted_number: str (just the number, no confidence)
    """
    model = create_ocr_model(model_name, device=device)
    number, _ = model.predict(bgr_image)
    return number


def create_ocr_model_number_only(
    model_name: Literal["resnet", "resnet18", "x2", "resnet34", "smol", "smolvlm2"],
    device: str = "auto",
):
    """
    Create an OCR model that returns only the number (no confidence).

    Args:
        model_name: Model identifier - one of "resnet", "x2", "smol"
        device: Device to run inference on ("auto", "cpu", "cuda", "mps")

    Returns:
        Model instance with predict(bgr_image) -> str interface
    """
    base_model = create_ocr_model(model_name, device=device)
    return NumberOnlyModel(base_model)
