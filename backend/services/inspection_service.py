import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from detector import run_full_pipeline
from schemas import InspectionParameters


BASE_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BACKEND_DIR / "outputs"
DB_PATH = BACKEND_DIR / "inspection_history.sqlite3"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def initialize_storage() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                inspection_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                num_defects INTEGER NOT NULL,
                total_defect_area REAL NOT NULL,
                image_area INTEGER NOT NULL,
                defect_percentage REAL NOT NULL,
                confidence REAL NOT NULL,
                verdict TEXT NOT NULL,
                inspection_duration_ms INTEGER NOT NULL,
                parameters TEXT NOT NULL,
                images TEXT NOT NULL
            )
            """
        )


def next_inspection_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    prefix = f"VI-{stamp}-"
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE inspection_id LIKE ?",
            (f"{prefix}%",),
        ).fetchone()
    return f"{prefix}{(row[0] or 0) + 1:04d}"


def validate_parameters(params: InspectionParameters) -> InspectionParameters:
    if params.canny_low >= params.canny_high:
        raise HTTPException(status_code=422, detail="canny_low must be lower than canny_high")
    if params.blur_kernel not in {3, 5, 7, 9, 11}:
        raise HTTPException(status_code=422, detail="blur_kernel must be one of 3, 5, 7, 9, 11")
    return params


async def decode_upload_to_bgr(image: UploadFile) -> tuple[np.ndarray, str]:
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="INVALID IMAGE INPUT")

    payload = await image.read()
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="INVALID IMAGE INPUT")

    try:
        pil_image = Image.open(__import__("io").BytesIO(payload))
        pil_image.verify()
        pil_image = Image.open(__import__("io").BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="CORRUPTED IMAGE INPUT")

    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), image.filename or "inspection_image.png"


def save_stage_image(path: Path, image: np.ndarray, grayscale: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if grayscale:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), image)


def inspect_bgr_image(
    bgr_image: np.ndarray,
    filename: str,
    params: InspectionParameters,
    save_results: bool = True,
) -> dict[str, Any]:
    validate_parameters(params)
    initialize_storage()
    inspection_id = next_inspection_id()
    started = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")

    result = run_full_pipeline(
        image=bgr_image,
        canny_thresh1=params.canny_low,
        canny_thresh2=params.canny_high,
        min_defect_area=params.min_defect_area,
        pass_fail_threshold=params.pass_fail_threshold,
        blur_kernel=params.blur_kernel,
        save_outputs=False,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)

    stage_dir = OUTPUT_DIR / inspection_id
    save_stage_image(stage_dir / "original.png", bgr_image)
    save_stage_image(stage_dir / "grayscale.png", result["grayscale"], grayscale=True)
    save_stage_image(stage_dir / "edges.png", result["edges"], grayscale=True)
    save_stage_image(stage_dir / "annotated.png", result["annotated"])

    height, width = bgr_image.shape[:2]
    images = {
        "original": f"/api/output/{inspection_id}/original.png",
        "grayscale": f"/api/output/{inspection_id}/grayscale.png",
        "edges": f"/api/output/{inspection_id}/edges.png",
        "annotated": f"/api/output/{inspection_id}/annotated.png",
    }
    response = {
        "inspection_id": inspection_id,
        "filename": filename,
        "image_width": width,
        "image_height": height,
        "inspection_timestamp": timestamp,
        "inspection_duration_ms": duration_ms,
        "parameters": params.model_dump(),
        "metrics": {
            "num_defects": int(result["num_defects"]),
            "total_defect_area": float(result["total_defect_area"]),
            "image_area": int(result["image_area"]),
            "defect_percentage": float(result["defect_percentage"]),
            "confidence": float(result["confidence"]),
        },
        "verdict": result["verdict"],
        "images": images,
    }

    if save_results:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inspection_id,
                    filename,
                    timestamp,
                    width,
                    height,
                    int(result["num_defects"]),
                    float(result["total_defect_area"]),
                    int(result["image_area"]),
                    float(result["defect_percentage"]),
                    float(result["confidence"]),
                    result["verdict"],
                    duration_ms,
                    json.dumps(params.model_dump()),
                    json.dumps(images),
                ),
            )
    return response


def list_samples() -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for category in ("good", "defective"):
        folder = DATASET_DIR / category
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                samples.append(
                    {
                        "filename": path.name,
                        "category": category.upper(),
                        "preview_url": f"/api/sample/{category}/{path.name}",
                    }
                )
    return samples


def get_history() -> list[dict[str, Any]]:
    initialize_storage()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM inspections ORDER BY timestamp DESC").fetchall()
    history = []
    for row in rows:
        item = dict(row)
        item["parameters"] = json.loads(item["parameters"])
        item["images"] = json.loads(item["images"])
        history.append(item)
    return history


def get_metrics() -> dict[str, float | int]:
    history = get_history()
    total = len(history)
    pass_count = sum(1 for item in history if item["verdict"] == "PASS")
    fail_count = sum(1 for item in history if item["verdict"] == "FAIL")
    coverage = sum(float(item["defect_percentage"]) for item in history)
    duration = sum(float(item["inspection_duration_ms"]) for item in history)
    return {
        "total_inspections": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round((pass_count / total) * 100, 2) if total else 0.0,
        "average_defect_coverage": round(coverage / total, 4) if total else 0.0,
        "average_inspection_time_ms": round(duration / total, 2) if total else 0.0,
    }


def reset_history() -> dict[str, str]:
    initialize_storage()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM inspections")
    return {"status": "reset"}

