# 🔬 VisionInspect AI — Automated Industrial Defect Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

**VisionInspect AI** is an open-source computer vision application that automatically detects surface defects in industrial products using classical image processing techniques. It provides a professional Streamlit dashboard where quality-control engineers can upload product images, tune detection parameters, and instantly receive a **PASS** or **FAIL** quality verdict — all without needing expensive ML infrastructure or labelled training data.

---

## 🚨 Problem Statement

Manual visual inspection of manufactured parts is:
- **Slow** — humans can only process so many parts per hour.
- **Inconsistent** — fatigue and subjectivity cause missed defects.
- **Costly** — dedicated inspection staff add significant overhead.

VisionInspect AI addresses this by providing a fast, reproducible, and configurable automated inspection pipeline powered by OpenCV.

---

## 🔄 Workflow

```
Input Image
    │
    ▼
Grayscale Conversion       ← Remove colour complexity
    │
    ▼
Gaussian Blur              ← Suppress noise
    │
    ▼
Canny Edge Detection       ← Find structural boundaries
    │
    ▼
Contour Detection          ← Identify closed regions
    │
    ▼
Area Filtering             ← Discard micro-noise contours
    │
    ▼
Defect Localisation        ← Red bounding boxes on defects
    │
    ▼
PASS / FAIL Classification ← Compare coverage % to threshold
```

---

## 🛠 Technologies Used

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.9+ | Core language |
| OpenCV | 4.9+ | Image processing engine |
| Streamlit | 1.32+ | Interactive web dashboard |
| NumPy | 1.26+ | Numerical array operations |
| Pillow | 10.2+ | Image I/O and format support |

---

## 📂 Project Structure

```
VisionInspect-AI/
│
├── app.py           ← Streamlit dashboard & UI
├── detector.py      ← Core CV pipeline (modular functions)
│
├── dataset/         ← Place your test images here
│   └── (add .jpg / .png product images)
│
├── outputs/         ← Saved annotated results (auto-created)
│   └── (annotated_YYYYMMDD_HHMMSS.png, ...)
│
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

---

## ⚙️ Installation Steps

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/VisionInspect-AI.git
cd VisionInspect-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## 🖥 Usage

1. Open the dashboard in your browser.
2. Navigate to **🏠 Home / Upload** in the sidebar.
3. Upload a product image (JPG, PNG, BMP, TIFF, or WebP).
4. Adjust detection parameters in the sidebar if needed:
   - **Lower / Upper Canny Threshold** — controls edge sensitivity.
   - **Min Defect Area** — filters out micro-noise contours.
   - **Pass/Fail Threshold (%)** — maximum allowed defect coverage.
   - **Blur Kernel Size** — controls preprocessing smoothing.
5. View the four pipeline stages and the PASS / FAIL verdict.
6. Check **📊 Analytics** for the detailed metrics table.

---

## 🎛 Adjustable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Lower Canny Threshold | 50 | Hysteresis lower bound |
| Upper Canny Threshold | 150 | Hysteresis upper bound |
| Min Defect Area | 100 px² | Minimum contour size |
| Pass/Fail Threshold | 2.0 % | Max defect coverage to pass |
| Blur Kernel Size | 5×5 | Gaussian preprocessing |

---

## 📊 Dashboard Outputs

- **Original Image** — raw uploaded product photo.
- **Grayscale Image** — luminance-only representation.
- **Edge Detection Image** — Canny binary edge map.
- **Annotated Result** — original image with red defect bounding boxes.
- **Quick Stats Cards** — defects found, total area, coverage %, confidence.
- **PASS / FAIL Badge** — clear green or red verdict.
- **Analytics Table** — full metrics breakdown.

---

## 🚀 Future Enhancements

- [ ] **Deep Learning Mode** — swap classical pipeline for a trained CNN (e.g. YOLOv8, EfficientDet) for higher accuracy on complex textures.
- [ ] **Batch Processing** — upload and inspect a folder of images at once with a CSV report export.
- [ ] **Defect Classification** — categorise defect types (scratch, crack, dent, stain) not just detect them.
- [ ] **Real-Time Camera Feed** — integrate with webcam or IP camera for live line inspection.
- [ ] **Inspection History** — persist results to a SQLite database with trend charts.
- [ ] **REST API** — expose the pipeline as a FastAPI endpoint for integration with factory MES/SCADA systems.
- [ ] **Multi-ROI Support** — define regions of interest on the product for zone-specific tolerance levels.
- [ ] **Calibration Mode** — learn baseline texture from known-good samples to reduce false positives.

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

*Built with ❤️ using Python, OpenCV & Streamlit.*
