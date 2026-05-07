"""Compare a fine-tuned YOLO checkpoint against the base COCO YOLO on the
basketball test split, after collapsing classes to a {person, ball} supercategory.

The fine-tuned model has 12 basketball-specific classes (player, referee,
ball_vis_full, ...). Base YOLO only knows "person" and "sports ball" from COCO,
so a direct mAP head-to-head isn't apples-to-apples. This script maps both
sides to {0: person, 1: ball}; fine-tune-only labels (number, rim) are dropped
from GT so neither model is penalized for not being able to predict them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


FINETUNING_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS_DIR = FINETUNING_DIR / "checkpoints"

# Fine-tuned class index -> super class id. Indices not listed (4=number,
# 11=rim) are intentionally dropped.
FINETUNED_TO_SUPER: dict[int, int] = {
    0: 1,   # ball-in-basket
    1: 1,   # ball_vis_full
    2: 1,   # ball_vis_partial
    3: 1,   # ball_vis_visible
    5: 0,   # player
    6: 0,   # player-in-possession
    7: 0,   # player-jump-shot
    8: 0,   # player-layup-dunk
    9: 0,   # player-shot-block
    10: 0,  # referee
}

COCO_TO_SUPER: dict[int, int] = {
    0: 0,   # person
    32: 1,  # sports ball
}

SUPER_NAMES = ["person", "ball"]


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int)
    boxes, classes = [], []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=int)


def remap(
    boxes: np.ndarray,
    classes: np.ndarray,
    mapping: dict[int, int],
    confs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(classes) == 0:
        empty_conf = np.zeros((0,), dtype=np.float32) if confs is not None else None
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int), empty_conf
    keep = np.fromiter((c in mapping for c in classes.tolist()), dtype=bool, count=len(classes))
    if not keep.any():
        empty_conf = np.zeros((0,), dtype=np.float32) if confs is not None else None
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int), empty_conf
    new_classes = np.fromiter((mapping[int(c)] for c in classes[keep].tolist()), dtype=int, count=int(keep.sum()))
    return boxes[keep], new_classes, (confs[keep] if confs is not None else None)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(
    preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    gts: list[tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    iou_thr: float = 0.5,
) -> tuple[list[dict], float, list[dict]]:
    records: list[list[tuple[float, int, int]]] = [[] for _ in range(num_classes)]
    n_gt = [0] * num_classes

    for (pb, pc, pf), (gb, gc) in zip(preds, gts):
        for c in range(num_classes):
            n_gt[c] += int((gc == c).sum())
            p_idx = np.where(pc == c)[0]
            g_idx = np.where(gc == c)[0]
            if len(p_idx) == 0:
                continue
            order = np.argsort(-pf[p_idx])
            p_idx = p_idx[order]
            matched = np.zeros(len(g_idx), dtype=bool)
            for pi in p_idx:
                if len(g_idx) == 0:
                    records[c].append((float(pf[pi]), 0, 1))
                    continue
                ious = box_iou(pb[pi], gb[g_idx])
                best = int(np.argmax(ious))
                if ious[best] >= iou_thr and not matched[best]:
                    matched[best] = True
                    records[c].append((float(pf[pi]), 1, 0))
                else:
                    records[c].append((float(pf[pi]), 0, 1))

    rows = []
    curves: list[dict] = []
    for c in range(num_classes):
        recs = sorted(records[c], key=lambda x: -x[0])
        if not recs:
            rows.append({"class": SUPER_NAMES[c], "ap50": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0,
                         "tp": 0, "fp": 0, "n_gt": n_gt[c]})
            curves.append({"class": SUPER_NAMES[c], "recall": np.zeros(0), "precision": np.zeros(0)})
            continue
        tps = np.cumsum([r[1] for r in recs]).astype(np.float64)
        fps = np.cumsum([r[2] for r in recs]).astype(np.float64)
        recall = tps / max(n_gt[c], 1)
        precision = tps / np.maximum(tps + fps, 1.0)
        ap = compute_ap(recall, precision)
        f1_curve = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        best_i = int(np.argmax(f1_curve))
        rows.append({
            "class": SUPER_NAMES[c],
            "ap50": ap,
            "p": float(precision[best_i]),
            "r": float(recall[best_i]),
            "f1": float(f1_curve[best_i]),
            "tp": int(tps[-1]),
            "fp": int(fps[-1]),
            "n_gt": n_gt[c],
        })
        curves.append({"class": SUPER_NAMES[c], "recall": recall, "precision": precision})
    map50 = float(np.mean([row["ap50"] for row in rows]))
    return rows, map50, curves


def extract_yolo_pred(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=np.float32))
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy().astype(np.float32)
    return boxes, cls, conf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs base YOLO on the {person, ball} supercategory.")
    parser.add_argument("--data", required=True, help="Path to data.yaml.")
    parser.add_argument("--weights", required=True, help="Fine-tuned .pt (e.g. runs/.../weights/best.pt).")
    parser.add_argument("--base-model", default="yolo11s.pt", help="Base COCO YOLO model (default: yolo11s.pt).")
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CHECKPOINTS_DIR),
                        help="Where to put auto-downloaded base weights if --base-model is a bare filename.")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split from data.yaml to evaluate on.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Low conf threshold for mAP-style eval.")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for TP matching (mAP@0.5).")
    parser.add_argument("--device", default=None, help="auto | cpu | mps | cuda index.")
    parser.add_argument("--max-images", type=int, default=0, help="Cap test images for a quick run (0 = all).")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write metrics.json/.txt and pr_curves.png. "
                             "Default: <run>/eval next to the weights.")
    parser.add_argument("--no-plot", action="store_true", help="Skip pr_curves.png.")
    return parser


def resolve_base_model(model: str, checkpoints_dir: Path) -> str:
    candidate = Path(model)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return model
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoints_dir / model)


def format_table(label: str, rows: list[dict], map50: float) -> str:
    lines = [f"=== {label} ===",
             f"{'class':<8} {'AP@0.5':>8} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'GT':>6}"]
    for row in rows:
        lines.append(f"{row['class']:<8} {row['ap50']:>8.4f} {row['p']:>8.4f} {row['r']:>8.4f} "
                     f"{row['f1']:>8.4f} {row['tp']:>6d} {row['fp']:>6d} {row['n_gt']:>6d}")
    lines.append(f"mAP@0.5: {map50:.4f}")
    return "\n".join(lines)


def format_delta(ft_rows: list[dict], ft_map: float, base_rows: list[dict], base_map: float) -> str:
    lines = ["=== delta (fine-tuned - base) ===",
             f"{'class':<8} {'dAP@0.5':>9} {'dP':>9} {'dR':>9} {'dF1':>9}"]
    for ftr, br in zip(ft_rows, base_rows):
        lines.append(f"{ftr['class']:<8} {ftr['ap50'] - br['ap50']:>+9.4f} "
                     f"{ftr['p'] - br['p']:>+9.4f} {ftr['r'] - br['r']:>+9.4f} "
                     f"{ftr['f1'] - br['f1']:>+9.4f}")
    lines.append(f"dmAP@0.5 : {ft_map - base_map:+.4f}")
    return "\n".join(lines)


def save_pr_curves(
    out_path: Path,
    ft_curves: list[dict],
    base_curves: list[dict],
    ft_label: str,
    base_label: str,
) -> None:
    import matplotlib.pyplot as plt

    n = len(ft_curves)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, ft, b in zip(axes[0], ft_curves, base_curves):
        if len(ft["recall"]):
            ax.plot(ft["recall"], ft["precision"], label=ft_label, linewidth=2)
        if len(b["recall"]):
            ax.plot(b["recall"], b["precision"], label=base_label, linewidth=2, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(ft["class"])
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("PR curves @ IoU 0.5 — fine-tuned vs base")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def default_out_dir(weights_path: Path) -> Path:
    """If weights live at <run>/weights/best.pt, default output to <run>/eval."""
    parent = weights_path.parent
    run_dir = parent.parent if parent.name == "weights" else parent
    return run_dir / "eval"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing dep. Run: uv sync --extra train") from exc

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text())
    dataset_root = data_yaml.parent
    split_rel = data.get(args.split)
    if split_rel is None:
        raise SystemExit(f"Split '{args.split}' not found in {data_yaml}")
    images_dir = (dataset_root / split_rel).resolve()
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Images dir not found: {images_dir}")

    image_paths = sorted(p for p in images_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images in {images_dir}")
    print(f"Evaluating {len(image_paths)} images from split '{args.split}'.")

    base_weights = resolve_base_model(args.base_model, Path(args.checkpoints_dir).resolve())
    finetuned = YOLO(args.weights)
    base = YOLO(base_weights)

    ft_preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    base_preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    gts: list[tuple[np.ndarray, np.ndarray]] = []

    predict_kwargs = dict(imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
    if args.device is not None:
        predict_kwargs["device"] = args.device

    for i, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes, gt_cls = load_yolo_labels(labels_dir / f"{img_path.stem}.txt", w, h)
        gt_boxes, gt_cls, _ = remap(gt_boxes, gt_cls, FINETUNED_TO_SUPER)
        gts.append((gt_boxes, gt_cls))

        r = finetuned.predict(img, **predict_kwargs)[0]
        b, c, f = extract_yolo_pred(r)
        b, c, f = remap(b, c, FINETUNED_TO_SUPER, f)
        ft_preds.append((b, c, f))

        r = base.predict(img, **predict_kwargs)[0]
        b, c, f = extract_yolo_pred(r)
        b, c, f = remap(b, c, COCO_TO_SUPER, f)
        base_preds.append((b, c, f))

        if i % 50 == 0 or i == len(image_paths):
            print(f"  processed {i}/{len(image_paths)}")

    ft_rows, ft_map, ft_curves = evaluate(ft_preds, gts, num_classes=len(SUPER_NAMES), iou_thr=args.match_iou)
    base_rows, base_map, base_curves = evaluate(base_preds, gts, num_classes=len(SUPER_NAMES), iou_thr=args.match_iou)

    ft_label = f"fine-tuned ({Path(args.weights).name})"
    base_label = f"base ({Path(base_weights).name})"
    sections = [
        format_table(ft_label, ft_rows, ft_map),
        format_table(base_label, base_rows, base_map),
        format_delta(ft_rows, ft_map, base_rows, base_map),
    ]
    report = "\n\n".join(sections)
    print()
    print(report)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(Path(args.weights).resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "config": {
            "data": str(data_yaml),
            "split": args.split,
            "weights": str(Path(args.weights).resolve()),
            "base_model": str(base_weights),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "match_iou": args.match_iou,
            "n_images": len(image_paths),
            "super_names": SUPER_NAMES,
        },
        "fine_tuned": {"per_class": ft_rows, "map50": ft_map},
        "base":       {"per_class": base_rows, "map50": base_map},
        "delta": {
            "per_class": [
                {"class": f["class"],
                 "ap50": f["ap50"] - b["ap50"],
                 "p":    f["p"]    - b["p"],
                 "r":    f["r"]    - b["r"],
                 "f1":   f["f1"]   - b["f1"]}
                for f, b in zip(ft_rows, base_rows)
            ],
            "map50": ft_map - base_map,
        },
    }
    import json
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "metrics.txt").write_text(report + "\n")
    print(f"\nWrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'metrics.txt'}")

    if not args.no_plot:
        try:
            save_pr_curves(out_dir / "pr_curves.png", ft_curves, base_curves, ft_label, base_label)
            print(f"Wrote {out_dir / 'pr_curves.png'}")
        except Exception as exc:
            print(f"Skipped pr_curves.png: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
