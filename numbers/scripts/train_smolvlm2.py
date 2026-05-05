from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoProcessor, Trainer, TrainingArguments

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

from jersey_numbers.ocr.smolvlm2 import DEFAULT_BASE_MODEL, resolve_device

_AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(degrees=10),
])


class JerseyNumberVQADataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: Any, augment: bool = False) -> None:
        with jsonl_path.open(encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
        self.augment = augment

        if augment:
            self._cache_dir = None
            self._prompt_len = self._compute_prompt_len()
        else:
            self._cache_dir = jsonl_path.parent / ".cache" / jsonl_path.stem
            self._build_cache()
            self.processor = None

    def _build_cache(self) -> None:
        if self._cache_dir.exists() and len(list(self._cache_dir.glob("*.pt"))) == len(self.samples):
            print(f"  Cache hit: {self._cache_dir}")
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Pre-tokenizing {len(self.samples)} samples → {self._cache_dir}")

        prompt_len = self._compute_prompt_len()
        for idx, entry in enumerate(tqdm(self.samples, desc="Caching")):
            image = Image.open(entry["image"]).convert("RGB")

            full_text = self.processor.apply_chat_template(
                [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": entry["prompt"]}]},
                    {"role": "assistant", "content": [{"type": "text", "text": entry["answer"]}]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            full_inputs = self.processor(text=full_text, images=[image], return_tensors="pt")

            input_ids = full_inputs["input_ids"].squeeze(0)
            attention_mask = full_inputs["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            sample = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            for key in full_inputs:
                if key not in ("input_ids", "attention_mask"):
                    sample[key] = full_inputs[key].squeeze(0)

            torch.save(sample, self._cache_dir / f"{idx:06d}.pt")

    def _compute_prompt_len(self) -> int:
        prompt = self.samples[0]["prompt"]
        prompt_text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        dummy = Image.new("RGB", (224, 224))
        return self.processor(text=prompt_text, images=[dummy], return_tensors="pt")["input_ids"].shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self._cache_dir is not None:
            return torch.load(self._cache_dir / f"{idx:06d}.pt", weights_only=True)

        entry = self.samples[idx]
        image = _AUGMENT(Image.open(entry["image"]).convert("RGB"))

        full_text = self.processor.apply_chat_template(
            [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": entry["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": entry["answer"]}]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        full_inputs = self.processor(text=full_text, images=[image], return_tensors="pt")

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[:self._prompt_len] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        for key in full_inputs:
            if key not in ("input_ids", "attention_mask"):
                result[key] = full_inputs[key].squeeze(0)

        return result


@dataclass
class VLMDataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch: dict[str, torch.Tensor] = {}

        for key, pad_val in [("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)]:
            seqs = [f[key] for f in features]
            max_len = max(s.shape[0] for s in seqs)
            padded = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
            for i, s in enumerate(seqs):
                padded[i, : s.shape[0]] = s
            batch[key] = padded

        for key in features[0]:
            if key not in batch:
                batch[key] = torch.stack([f[key] for f in features])

        return batch


def _detect_precision(device: torch.device) -> tuple[bool, bool]:
    if device.type != "cuda":
        return False, False
    bf16 = torch.cuda.is_bf16_supported()
    fp16 = not bf16
    return bf16, fp16


def train(
    jsonl_dir: Path,
    output_dir: Path,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 2,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    lr: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    device_str: str = "auto",
) -> None:
    device = resolve_device(device_str)
    print(f"Using device: {device}")

    bf16, fp16 = _detect_precision(device)
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    print(f"Precision: {'bf16' if bf16 else 'fp16' if fp16 else 'fp32'}")

    transformers.logging.set_verbosity_error()
    logging.getLogger("PIL").setLevel(logging.WARNING)

    print(f"Loading processor and base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForVision2Seq.from_pretrained(base_model, torch_dtype=dtype)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building datasets...")
    train_ds = JerseyNumberVQADataset(jsonl_dir / "train.jsonl", processor, augment=True)
    val_ds = JerseyNumberVQADataset(jsonl_dir / "val.jsonl", processor, augment=False)
    print(f"  train: {len(train_ds)} samples  val: {len(val_ds)} samples")
    print(f"  effective batch size: {batch_size * grad_accum_steps}")

    known_classes = sorted({entry["answer"] for entry in train_ds.samples})

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        disable_tqdm=False,
    )

    collator = VLMDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"\nSaving checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    class_list_path = output_dir / "class_list.json"
    with class_list_path.open("w") as f:
        json.dump(known_classes, f)
    print(f"Saved {len(known_classes)} classes → {class_list_path}")

    print("Done.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM2 on jersey number crops.")
    parser.add_argument("--data", required=True, help="JSONL directory (output of prepare_dataset_smolvlm2.py).")
    parser.add_argument("--output", required=True, help="Checkpoint output directory.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HuggingFace base model name.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps (effective batch = batch-size * grad-accum-steps).")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train(
        jsonl_dir=Path(args.data),
        output_dir=Path(args.output),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
