from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Instruct"
PROMPT = "Read the jersey number."


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SmolVLM2OCR:
    """Load a fine-tuned SmolVLM2 LoRA checkpoint and predict jersey numbers from BGR crops."""

    def __init__(self, checkpoint_path: Path, device_str: str = "auto") -> None:
        from peft import PeftModel
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
        from transformers import AutoProcessor

        self.device = resolve_device(device_str)
        checkpoint_path = Path(checkpoint_path)

        self.processor = AutoProcessor.from_pretrained(checkpoint_path)

        class_list_path = checkpoint_path / "class_list.json"
        self.known_classes: set[str] = set()
        if class_list_path.exists():
            with class_list_path.open() as f:
                self.known_classes = set(json.load(f))

        base_model_name = _load_base_model_name(checkpoint_path)
        dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        base = AutoModelForVision2Seq.from_pretrained(base_model_name, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, checkpoint_path).to(self.device).eval()

    def predict(self, bgr_image: np.ndarray) -> tuple[str, float]:
        image = Image.fromarray(bgr_image[..., ::-1].copy()).resize((224, 224))

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = out.sequences[0][prompt_len:]
        raw = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        match = re.search(r"\d+", raw)
        label = match.group().lstrip("0") or "0" if match else "unknown"

        confidence = 1.0
        if out.scores:
            probs = torch.softmax(out.scores[0][0], dim=-1)
            confidence = float(probs.max().item())

        return label, confidence

    def predict_batch(self, bgr_images: list[np.ndarray]) -> list[tuple[str, float]]:
        return [self.predict(img) for img in bgr_images]


def _load_base_model_name(checkpoint_path: Path) -> str:
    adapter_config = checkpoint_path / "adapter_config.json"
    if adapter_config.exists():
        with adapter_config.open() as f:
            base = json.load(f).get("base_model_name_or_path")
        if base:
            return base
    return DEFAULT_BASE_MODEL
