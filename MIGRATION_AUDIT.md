# VisionInspect AI Migration Audit

## Current Pipeline

The existing app was a Streamlit dashboard that imported `run_full_pipeline` and `save_output_image` from `detector.py`. The detector is deterministic OpenCV code, not a trained model.

Pipeline stages:

1. Load BGR image from a file path or NumPy array.
2. Convert BGR image to grayscale.
3. Apply Gaussian blur with an odd kernel.
4. Run Canny edge detection.
5. Find external contours from the edge map.
6. Filter contours by minimum area.
7. Draw red bounding boxes and labels on the original image.
8. Calculate defect count, total contour area, image area, and coverage percentage.
9. Compare defect coverage against the configured pass/fail threshold.
10. Optionally save annotated and edge images.

## Detector Inputs

`run_full_pipeline(image, canny_thresh1=50, canny_thresh2=150, min_defect_area=100, pass_fail_threshold=2.0, blur_kernel=5, save_outputs=False)`

`image` is expected to be a NumPy array in OpenCV BGR format.

## Detector Outputs

Verified by running the current detector against:

- `dataset/good/pcb_good_1.png`
- `dataset/defective/pcb_defect_1.png`

Return keys:

- `grayscale`: `uint8` ndarray, shape `(height, width)`
- `blurred`: `uint8` ndarray, shape `(height, width)`
- `edges`: `uint8` ndarray, shape `(height, width)`
- `contours`: list of OpenCV contour arrays
- `annotated`: `uint8` ndarray, shape `(height, width, 3)` in BGR
- `num_defects`: int
- `total_defect_area`: float
- `image_area`: int
- `defect_percentage`: float rounded to 4 decimals
- `verdict`: `PASS`, `FAIL`, or fallback `UNKNOWN`
- `confidence`: float rounded to 2 decimals
- `saved_annotated`: present only when `save_outputs=True`
- `saved_edges`: present only when `save_outputs=True`

Baseline default results:

| Image | Size | Defects | Area | Coverage | Verdict | Confidence |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `dataset/good/pcb_good_1.png` | 1024 x 1024 | 1 | 841675.0 | 80.2684% | FAIL | 100.0 |
| `dataset/defective/pcb_defect_1.png` | 1024 x 1024 | 1 | 841675.0 | 80.2684% | FAIL | 100.0 |

These results are preserved as existing detector behavior.

## Current Parameters

- Lower Canny threshold: default `50`, Streamlit range `10..200`
- Upper Canny threshold: default `150`, Streamlit range `50..400`
- Minimum defect area: default `100`, Streamlit range `10..2000`
- Pass/fail threshold: default `2.0%`, Streamlit range `0.1..20.0`
- Blur kernel size: default `5`, options `3, 5, 7, 9, 11`
- Save outputs: default `true` in Streamlit UI

## Current Pass/Fail Logic

`classify_product` returns `PASS` when `defect_percentage <= pass_fail_threshold`; otherwise it returns `FAIL`.

Confidence is a threshold-distance score:

- PASS: `50 + ((threshold - coverage) / threshold) * 50`, capped at `100`
- FAIL: `50 + ((coverage - threshold) / threshold) * 50`, capped at `100`

This is not an AI probability.

## Files To Preserve

- `backend/detector.py`: single source of truth for OpenCV inspection
- `dataset/good/*`
- `dataset/defective/*`
- `generate_sample.py`
- Existing `outputs/` artifacts are left in place; new API artifacts are written to `backend/outputs/`

## Migration Plan

1. Move the detector into `backend/detector.py` without changing detection behavior.
2. Add FastAPI endpoints for health, inspection, samples, history, metrics, and safe media serving.
3. Serialize real pipeline stage images as PNG files under `backend/outputs/{inspection_id}`.
4. Persist lightweight inspection history in SQLite for analytics and history views.
5. Build a React + TypeScript + Vite dashboard with an industrial inspection terminal UI.
6. Keep all verdicts and metrics backend-owned.
7. Validate samples, parameters, stage images, history, analytics, type checking, and production build.

