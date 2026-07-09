"""
app.py - VisionInspect AI Streamlit Dashboard
Main application entry point for the industrial defect detection system.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Import our custom detection module
from detector import run_full_pipeline, save_output_image

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VisionInspect AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Industrial Theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global background ── */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 2px solid #2e3250;
    }
    section[data-testid="stSidebar"] * {
        color: #c8cde4 !important;
    }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #2962ff;
        box-shadow: 0 4px 20px rgba(41, 98, 255, 0.3);
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .header-banner p {
        color: #90caf9;
        margin: 6px 0 0 0;
        font-size: 1rem;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: #1e2235;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .stat-card .stat-label {
        color: #90a4ae;
        font-size: 0.78rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .stat-card .stat-value {
        color: #e3f2fd;
        font-size: 1.7rem;
        font-weight: 700;
    }
    .stat-card .stat-unit {
        color: #78909c;
        font-size: 0.8rem;
    }

    /* ── PASS / FAIL badge ── */
    .badge-pass {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: #a5d6a7;
        border: 2px solid #43a047;
        border-radius: 50px;
        padding: 10px 40px;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 4px;
        display: inline-block;
        box-shadow: 0 0 20px rgba(67, 160, 71, 0.5);
    }
    .badge-fail {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        color: #ffcdd2;
        border: 2px solid #ef5350;
        border-radius: 50px;
        padding: 10px 40px;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 4px;
        display: inline-block;
        box-shadow: 0 0 20px rgba(239, 83, 80, 0.5);
    }

    /* ── Section headings ── */
    .section-title {
        color: #42a5f5;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #2e3250;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* ── Image captions ── */
    .img-caption {
        text-align: center;
        color: #78909c;
        font-size: 0.8rem;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }

    /* ── Info box ── */
    .info-box {
        background: #1a237e22;
        border-left: 4px solid #1565c0;
        border-radius: 6px;
        padding: 12px 16px;
        color: #90caf9;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }

    /* ── Divider ── */
    hr { border-color: #2e3250; }

    /* ── Streamlit image labels ── */
    .stImage > div > div > div {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def pil_to_bgr(pil_image):
    """Convert a PIL Image to a BGR numpy array (OpenCV format)."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr_image):
    """Convert a BGR numpy array to RGB for Streamlit display."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def gray_to_rgb(gray_image):
    """Convert a grayscale numpy array to RGB for Streamlit display."""
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def ensure_output_dir():
    """Make sure the outputs directory exists."""
    os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar navigation and parameter controls."""

    with st.sidebar:
        st.markdown("## 🔬 VisionInspect AI")
        st.markdown("*Industrial Quality Control*")
        st.markdown("---")

        st.markdown("### 📌 Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home / Upload", "📊 Analytics", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Detection Parameters")

        st.markdown("**Edge Detection Thresholds**")
        canny_low = st.slider(
            "Lower Threshold",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Lower bound for Canny hysteresis. Edges weaker than this are discarded.",
        )
        canny_high = st.slider(
            "Upper Threshold",
            min_value=50,
            max_value=400,
            value=150,
            step=10,
            help="Upper bound for Canny hysteresis. Edges stronger than this are always kept.",
        )

        st.markdown("**Contour Filtering**")
        min_area = st.slider(
            "Min Defect Area (px²)",
            min_value=10,
            max_value=2000,
            value=100,
            step=10,
            help="Contours smaller than this area are ignored as noise.",
        )

        st.markdown("**Quality Gate**")
        pf_threshold = st.slider(
            "Pass/Fail Threshold (%)",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Maximum allowed defect coverage (%) for a product to PASS.",
        )

        st.markdown("**Preprocessing**")
        blur_kernel = st.select_slider(
            "Blur Kernel Size",
            options=[3, 5, 7, 9, 11],
            value=5,
            help="Gaussian blur kernel — larger values reduce more noise but may blur fine defects.",
        )

        save_results = st.checkbox(
            "💾 Save results to /outputs",
            value=True,
            help="Automatically save annotated output images.",
        )

        st.markdown("---")
        st.markdown(
            "<div style='color:#546e7a;font-size:0.75rem;text-align:center;'>"
            "VisionInspect AI v1.0<br>© 2024 Industrial CV</div>",
            unsafe_allow_html=True,
        )

    return {
        "page": page,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "min_area": min_area,
        "pf_threshold": pf_threshold,
        "blur_kernel": blur_kernel,
        "save_results": save_results,
    }


# ─────────────────────────────────────────────
# STAT CARDS
# ─────────────────────────────────────────────

def render_stat_cards(metrics, verdict, confidence):
    """Render the four analytics stat cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-label">Defects Found</div>
                <div class="stat-value">{metrics['num_defects']}</div>
                <div class="stat-unit">contours</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-label">Total Defect Area</div>
                <div class="stat-value">{int(metrics['total_defect_area']):,}</div>
                <div class="stat-unit">pixels²</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-label">Defect Coverage</div>
                <div class="stat-value">{metrics['defect_percentage']:.3f}</div>
                <div class="stat-unit">%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="stat-value">{confidence:.1f}</div>
                <div class="stat-unit">%</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# IMAGE PIPELINE DISPLAY
# ─────────────────────────────────────────────

def render_pipeline_images(original_bgr, results):
    """Display the four pipeline stages in a 2×2 grid."""
    st.markdown('<p class="section-title">📷 Processing Pipeline</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.image(bgr_to_rgb(original_bgr), use_container_width=True, caption="")
        st.markdown('<p class="img-caption">① Original Image</p>', unsafe_allow_html=True)

    with col2:
        st.image(gray_to_rgb(results["grayscale"]), use_container_width=True, caption="")
        st.markdown('<p class="img-caption">② Grayscale Conversion</p>', unsafe_allow_html=True)

    with col3:
        st.image(gray_to_rgb(results["edges"]), use_container_width=True, caption="")
        st.markdown('<p class="img-caption">③ Canny Edge Detection</p>', unsafe_allow_html=True)

    with col4:
        st.image(bgr_to_rgb(results["annotated"]), use_container_width=True, caption="")
        st.markdown(
            '<p class="img-caption">④ Annotated Result (defects in red)</p>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# VERDICT BANNER
# ─────────────────────────────────────────────

def render_verdict(verdict, confidence, pf_threshold):
    """Render the PASS / FAIL verdict banner."""
    st.markdown("---")
    st.markdown('<p class="section-title">🏆 Quality Inspection Result</p>', unsafe_allow_html=True)

    v_col, i_col = st.columns([1, 2])

    with v_col:
        badge_class = "badge-pass" if verdict == "PASS" else "badge-fail"
        icon = "✅" if verdict == "PASS" else "❌"
        st.markdown(
            f'<div style="text-align:center;padding:20px 0;">'
            f'<span class="{badge_class}">{icon} {verdict}</span></div>',
            unsafe_allow_html=True,
        )

    with i_col:
        status_color = "#a5d6a7" if verdict == "PASS" else "#ef9a9a"
        st.markdown(
            f"""
            <div class="info-box" style="border-color:{'#43a047' if verdict=='PASS' else '#ef5350'};">
            <b style="color:{status_color};">Inspection Summary</b><br><br>
            {'✔ Product defect coverage is within acceptable limits.' if verdict == 'PASS'
             else '✘ Product defect coverage exceeds the quality threshold.'}<br><br>
            <b>Pass/Fail Threshold:</b> {pf_threshold:.1f}%<br>
            <b>Detection Confidence:</b> {confidence:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# ANALYTICS PAGE
# ─────────────────────────────────────────────

def render_analytics_page(metrics, verdict, confidence, pf_threshold):
    """Render the dedicated analytics panel."""
    st.markdown('<p class="section-title">📊 Analytics Panel</p>', unsafe_allow_html=True)

    # Progress bar for defect coverage
    st.markdown("**Defect Coverage vs Threshold**")
    pct = min(metrics["defect_percentage"] / max(pf_threshold * 2, 0.001), 1.0)
    bar_color = "#ef5350" if verdict == "FAIL" else "#43a047"
    st.markdown(
        f"""
        <div style="background:#1e2235;border-radius:6px;height:24px;width:100%;margin-bottom:8px;">
            <div style="background:{bar_color};width:{pct*100:.1f}%;height:100%;
                        border-radius:6px;display:flex;align-items:center;
                        padding-left:8px;color:#fff;font-size:0.8rem;font-weight:600;">
                {metrics['defect_percentage']:.3f}%
            </div>
        </div>
        <p style="color:#546e7a;font-size:0.8rem;margin-top:-4px;">
            Threshold: {pf_threshold:.1f}%
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Detailed metrics table
    data = {
        "Metric": [
            "Number of Defects",
            "Total Defect Area",
            "Image Area",
            "Defect Coverage",
            "Pass/Fail Threshold",
            "Inspection Result",
            "Confidence Score",
        ],
        "Value": [
            str(metrics["num_defects"]),
            f"{int(metrics['total_defect_area']):,} px²",
            f"{int(metrics['image_area']):,} px²",
            f"{metrics['defect_percentage']:.4f} %",
            f"{pf_threshold:.1f} %",
            verdict,
            f"{confidence:.2f} %",
        ],
    }

    st.table(data)


# ─────────────────────────────────────────────
# ABOUT PAGE
# ─────────────────────────────────────────────

def render_about_page():
    """Render the About / Info page."""
    st.markdown('<p class="section-title">ℹ️ About VisionInspect AI</p>', unsafe_allow_html=True)
    st.markdown(
        """
        **VisionInspect AI** is an open-source computer vision application designed for
        automated industrial quality control. It uses classical image processing techniques
        to detect surface defects on manufactured parts without the need for expensive
        dedicated hardware or deep learning infrastructure.

        ---

        ### 🔄 Detection Workflow

        | Step | Operation | Purpose |
        |------|-----------|---------|
        | 1 | Grayscale Conversion | Reduce colour complexity |
        | 2 | Gaussian Blur | Suppress high-frequency noise |
        | 3 | Canny Edge Detection | Locate structural boundaries |
        | 4 | Contour Detection | Identify closed regions |
        | 5 | Area Filtering | Remove micro-noise |
        | 6 | Bounding Box Annotation | Localise defect positions |
        | 7 | PASS / FAIL Classification | Quality gate decision |

        ---

        ### 🛠 Technologies
        - **Python 3.9+**
        - **OpenCV** — image processing engine
        - **Streamlit** — interactive web dashboard
        - **NumPy** — numerical array operations
        - **Pillow** — image I/O

        ---

        ### 📂 Project Structure
        ```
        VisionInspect-AI/
        ├── app.py           ← Streamlit dashboard (this file)
        ├── detector.py      ← Core CV pipeline
        ├── dataset/         ← Sample / input images
        ├── outputs/         ← Saved annotated results
        ├── requirements.txt
        └── README.md
        ```

        ---

        ### 💡 Tuning Tips
        - **Lower Canny thresholds** → detect more edges (may add noise).
        - **Higher min defect area** → ignore small surface marks.
        - **Lower Pass/Fail threshold** → stricter quality gate.
        """
    )


# ─────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────

def main():
    ensure_output_dir()

    # Render sidebar and capture parameters
    params = render_sidebar()

    # ── Header banner ──────────────────────────
    st.markdown(
        """
        <div class="header-banner">
            <h1>🔬 VisionInspect AI</h1>
            <p>Automated Industrial Defect Detection System — Powered by OpenCV</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Page routing ───────────────────────────
    page = params["page"]

    # ── HOME PAGE ──────────────────────────────
    if page == "🏠 Home / Upload":

        st.markdown('<p class="section-title">📤 Upload Product Image</p>', unsafe_allow_html=True)

        upload_col, hint_col = st.columns([2, 1])
        with upload_col:
            uploaded_file = st.file_uploader(
                "Drag & drop or browse for an image",
                type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
                help="Supported formats: JPG, PNG, BMP, TIFF, WebP",
            )
        with hint_col:
            st.markdown(
                """
                <div class="info-box">
                <b>Supported content:</b><br>
                • PCB boards<br>
                • Metal surfaces<br>
                • Fabric / textile<br>
                • Ceramic / glass<br>
                • Any flat surface
                </div>
                """,
                unsafe_allow_html=True,
            )

        if uploaded_file is not None:
            # ── Load image ──────────────────────
            try:
                pil_img = Image.open(uploaded_file)
                bgr_img = pil_to_bgr(pil_img)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                return

            st.success(
                f"✅ Loaded: **{uploaded_file.name}** "
                f"({pil_img.width}×{pil_img.height} px)"
            )

            # ── Run pipeline ────────────────────
            with st.spinner("🔍 Analysing image for defects…"):
                results = run_full_pipeline(
                    image=bgr_img,
                    canny_thresh1=params["canny_low"],
                    canny_thresh2=params["canny_high"],
                    min_defect_area=params["min_area"],
                    pass_fail_threshold=params["pf_threshold"],
                    blur_kernel=params["blur_kernel"],
                    save_outputs=params["save_results"],
                )

            # ── Stat cards ──────────────────────
            st.markdown("---")
            st.markdown('<p class="section-title">📈 Quick Stats</p>', unsafe_allow_html=True)
            render_stat_cards(results, results["verdict"], results["confidence"])

            # ── Pipeline images ─────────────────
            st.markdown("---")
            render_pipeline_images(bgr_img, results)

            # ── Verdict ─────────────────────────
            render_verdict(results["verdict"], results["confidence"], params["pf_threshold"])

            # ── Save notification ───────────────
            if params["save_results"] and results.get("saved_annotated"):
                st.info(f"💾 Annotated result saved to: `{results['saved_annotated']}`")

        else:
            # Placeholder when no image is uploaded
            st.markdown(
                """
                <div style="
                    background:#1e2235;border:2px dashed #2e3250;border-radius:12px;
                    padding:60px;text-align:center;margin-top:16px;">
                    <span style="font-size:3rem;">🏭</span><br><br>
                    <span style="color:#546e7a;font-size:1rem;">
                        Upload a product image to start the inspection
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── ANALYTICS PAGE ─────────────────────────
    elif page == "📊 Analytics":
        st.info(
            "Run an inspection from the **Home / Upload** page first, "
            "then return here to see detailed analytics.",
            icon="ℹ️",
        )
        # Show demo analytics if session has results cached
        if "last_results" in st.session_state:
            r = st.session_state["last_results"]
            render_analytics_page(r, r["verdict"], r["confidence"], params["pf_threshold"])

    # ── ABOUT PAGE ─────────────────────────────
    elif page == "ℹ️ About":
        render_about_page()


if __name__ == "__main__":
    main()
