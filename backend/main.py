from pathlib import Path
import os

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from schemas import HealthResponse, InspectionParameters, InspectionResponse, MetricsResponse, SamplesResponse
from services.inspection_service import (
    DATASET_DIR,
    OUTPUT_DIR,
    decode_upload_to_bgr,
    get_history,
    get_metrics,
    initialize_storage,
    inspect_bgr_image,
    list_samples,
    reset_history,
)


app = FastAPI(title="VisionInspect AI API", version="2.0")

# Allow origins from env var (comma-separated) or fall back to localhost dev
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_storage()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/inspect", response_model=InspectionResponse)
async def inspect(
    image: UploadFile = File(...),
    canny_low: int = Form(50),
    canny_high: int = Form(150),
    min_defect_area: int = Form(50),
    pass_fail_threshold: float = Form(1.0),
    blur_kernel: int = Form(5),
    save_results: bool = Form(True),
) -> dict:
    params = InspectionParameters(
        canny_low=canny_low,
        canny_high=canny_high,
        min_defect_area=min_defect_area,
        pass_fail_threshold=pass_fail_threshold,
        blur_kernel=blur_kernel,
    )
    bgr_image, filename = await decode_upload_to_bgr(image)
    return inspect_bgr_image(bgr_image, filename, params, save_results)


@app.get("/api/samples", response_model=SamplesResponse)
def samples() -> dict:
    return {"samples": list_samples()}


@app.get("/api/history")
def history() -> list[dict]:
    return get_history()


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> dict:
    return get_metrics()


@app.post("/api/reset-history")
def reset() -> dict:
    return reset_history()


@app.get("/api/output/{inspection_id}/{filename}")
def output_file(inspection_id: str, filename: str) -> FileResponse:
    if "/" in inspection_id or "\\" in inspection_id or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="OUTPUT ARTIFACT UNAVAILABLE")
    path = OUTPUT_DIR / inspection_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="OUTPUT ARTIFACT UNAVAILABLE")
    return FileResponse(path)


@app.get("/api/sample/{category}/{filename}")
def sample_file(category: str, filename: str) -> FileResponse:
    normalized = category.lower()
    if normalized not in {"good", "defective"} or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="INVALID SAMPLE")
    path = DATASET_DIR / normalized / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="INVALID SAMPLE")
    return FileResponse(path)


@app.get("/api/sample-inspect/{category}/{filename}", response_model=InspectionResponse)
def inspect_sample(
    category: str,
    filename: str,
    canny_low: int = 50,
    canny_high: int = 150,
    min_defect_area: int = 50,
    pass_fail_threshold: float = 1.0,
    blur_kernel: int = 5,
    save_results: bool = True,
) -> dict:
    normalized = category.lower()
    if normalized not in {"good", "defective"} or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="INVALID SAMPLE")
    path = DATASET_DIR / normalized / filename
    bgr_image = cv2.imread(str(path))
    if bgr_image is None:
        raise HTTPException(status_code=404, detail="INVALID SAMPLE")
    params = InspectionParameters(
        canny_low=canny_low,
        canny_high=canny_high,
        min_defect_area=min_defect_area,
        pass_fail_threshold=pass_fail_threshold,
        blur_kernel=blur_kernel,
    )
    return inspect_bgr_image(bgr_image, filename, params, save_results)

