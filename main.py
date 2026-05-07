"""Main entry point for cv-group pipeline.

This module demonstrates how to use the OCR model factory for jersey number recognition.
The object detection pipeline provides bounding boxes, then matching associates numbers
with players, and finally OCR predicts the actual jersey numbers.

Usage:
    PYTHONPATH=./numbers:$PYTHONPATH python main.py
"""

from __future__ import annotations

import numpy as np

# Import after PYTHONPATH is set (see module docstring)
from jersey_numbers.ocr.model_factory import create_ocr_model


def run_ocr_on_crop(bgr_crop: np.ndarray, model_name: str = "resnet") -> tuple[str, float]:
    """
    Run OCR on a jersey number crop.

    This is called after object detection provides the bounding box
    and matching associates it with a player track.

    Args:
        bgr_crop: Cropped image of the jersey number (BGR format from OpenCV)
        model_name: Which model to use - "resnet", "x2", or "smol"
                   (hardcoded - change this value to switch models)

    Returns:
        (jersey_number: str, confidence: float)

    Example:
        >>> import cv2
        >>> crop = cv2.imread("number_crop.jpg")  # BGR image
        >>> number, conf = run_ocr_on_crop(crop, model_name="resnet")
        >>> print(f"Player wears number {number} (confidence: {conf:.2f})")
    """
    model = create_ocr_model(model_name)
    return model.predict(bgr_crop)


def main():
    """Main entry point."""
    print("CV-Group Basketball Computer Vision Pipeline")
    print("=" * 50)

    # Example: How to integrate OCR into the detection pipeline
    # Hardcode your model choice here - edit to switch models
    OCR_MODEL_NAME = "resnet"  # Options: "resnet", "x2", "smol"

    print(f"\nUsing OCR model: {OCR_MODEL_NAME}")
    print("\nTo use in your pipeline:")
    print("  1. Import: from numbers.jersey_numbers.ocr.model_factory import create_ocr_model")
    print("  2. Create model: ocr_model = create_ocr_model('resnet')  # or 'x2' or 'smol'")
    print("  3. Predict: number, confidence = ocr_model.predict(bgr_crop)")
    print("\nOr use with the matching module:")
    print("  from numbers.jersey_numbers.matching.ios import match_frame")
    print("  matches = match_frame(")
    print("      frame_id=frame_id,")
    print("      number_detections=detections,")
    print("      tracks=player_tracks,")
    print("      ios_threshold=0.9,")
    print("      frame_image=frame,")
    print("      ocr_model=ocr_model,")
    print("  )")


if __name__ == "__main__":
    main()
