from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.utils import ensure_dir
from src.yolo_client import YoloLocalClient, YoloLocalSettings

from .config import ApiSettings
from .pipeline import ProcessResult, process_video


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    settings = ApiSettings.from_env()
    ensure_dir(settings.work_dir)

    yolo = YoloLocalClient(
        YoloLocalSettings(
            weights_path=str(settings.yolo_weights),
            imgsz=settings.yolo_imgsz,
            conf=settings.yolo_conf,
            iou=settings.yolo_iou,
            device=settings.yolo_device,
        )
    )

    sam2_predictor: Any | None = None
    if settings.sam2_enabled:
        sam2_predictor = _build_sam2_predictor(settings)

    app.state.settings = settings
    app.state.yolo = yolo
    app.state.sam2_predictor = sam2_predictor

    yield


app = FastAPI(title="Basketball CV API", lifespan=lifespan, redirect_slashes=False)


@app.get("/healthz")
def healthz(request: Request) -> dict[str, Any]:
    settings: ApiSettings = request.app.state.settings
    return {
        "status": "ok",
        "yolo_weights": str(settings.yolo_weights),
        "sam2_enabled": settings.sam2_enabled,
    }


@app.post("/process")
async def process(request: Request, video: UploadFile = File(...)) -> Any:
    settings: ApiSettings = request.app.state.settings
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    upload_path = await _save_upload(video, settings)

    try:
        result: ProcessResult = process_video(
            video_path=upload_path,
            yolo=request.app.state.yolo,
            sam2_predictor=request.app.state.sam2_predictor,
            settings=settings,
        )
    finally:
        upload_path.unlink(missing_ok=True)

    return FileResponse(
        path=str(result.annotated_video_path),
        media_type="video/mp4",
        filename=f"{result.job_id}.mp4",
    )


async def _save_upload(upload: UploadFile, settings: ApiSettings) -> Path:
    ensure_dir(settings.work_dir)
    suffix = Path(upload.filename or "").suffix.lower() or ".mp4"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with tempfile.NamedTemporaryFile(
        suffix=suffix, dir=str(settings.work_dir), delete=False
    ) as tmp:
        copied = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_bytes:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {settings.max_upload_mb} MB limit.",
                )
            tmp.write(chunk)
        return Path(tmp.name)


def _build_sam2_predictor(settings: ApiSettings) -> Any:
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 enabled but the 'sam2' package is not installed. "
            "Install it (see object-detection/README.md) or set BASKETBALL_API_SAM2_ENABLED=0."
        ) from exc
    if settings.sam2_checkpoint is None or not settings.sam2_checkpoint.exists():
        raise RuntimeError(f"SAM2 checkpoint not found: {settings.sam2_checkpoint}")
    device = _resolve_sam2_device(settings.sam2_device)
    return build_sam2_video_predictor(
        settings.sam2_model_cfg, str(settings.sam2_checkpoint), device=device
    )


def _resolve_sam2_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run() -> None:
    """Entry point for `uv run basketball-api`."""
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
