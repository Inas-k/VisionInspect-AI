"""
detector.py - Core defect detection engine for VisionInspect AI
Handles all image processing and defect analysis logic.
"""

import cv2
import numpy as np
import os
from datetime import datetime


def load_image(image_source):
    """
    Load an image from a file path or numpy array.

    Args:
        image_source: File path (str) or numpy array

    Returns:
        numpy array (BGR image) or None on failure
    """
    try:
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not read image from path: {image_source}")
            return img
        elif isinstance(image_source, np.ndarray):
            return image_source
        else:
            raise TypeError("image_source must be a file path or numpy array")
    except Exception as e:
        print(f"[ERROR] load_image: {e}")
        return None


def convert_to_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Args:
        image: numpy array (BGR)

    Returns:
        numpy array (grayscale)
    """
    try:
        if len(image.shape) == 2:
            # Already grayscale
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"[ERROR] convert_to_grayscale: {e}")
        return None


def apply_gaussian_blur(gray_image, kernel_size=5):
    """
    Apply Gaussian blur to reduce noise before edge detection.

    Args:
        gray_image: numpy array (grayscale)
        kernel_size: int, must be odd (default 5)

    Returns:
        numpy array (blurred grayscale image)
    """
    try:
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        return blurred
    except Exception as e:
        print(f"[ERROR] apply_gaussian_blur: {e}")
        return None


def apply_canny_edge_detection(blurred_image, threshold1=50, threshold2=150):
    """
    Apply Canny edge detection to find edges in the image.

    Args:
        blurred_image: numpy array (blurred grayscale)
        threshold1: int, lower threshold for hysteresis (default 50)
        threshold2: int, upper threshold for hysteresis (default 150)

    Returns:
        numpy array (binary edge map)
    """
    try:
        edges = cv2.Canny(blurred_image, threshold1, threshold2)
        return edges
    except Exception as e:
        print(f"[ERROR] apply_canny_edge_detection: {e}")
        return None


def find_defect_contours(edge_image, min_defect_area=100):
    """
    Find contours from the edge image and filter by minimum area.

    Args:
        edge_image: numpy array (binary edge map)
        min_defect_area: int, minimum contour area to count as a defect (default 100)

    Returns:
        list of contours that exceed the minimum area threshold
    """
    try:
        contours, _ = cv2.findContours(
            edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter contours by minimum area
        defect_contours = [c for c in contours if cv2.contourArea(c) >= min_defect_area]
        return defect_contours
    except Exception as e:
        print(f"[ERROR] find_defect_contours: {e}")
        return []


def annotate_defects(original_image, defect_contours):
    """
    Draw red bounding boxes around detected defects on the original image.

    Args:
        original_image: numpy array (BGR)
        defect_contours: list of contours

    Returns:
        numpy array (annotated BGR image)
    """
    try:
        annotated = original_image.copy()
        for i, contour in enumerate(defect_contours):
            x, y, w, h = cv2.boundingRect(contour)
            # Draw red bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Label each defect
            cv2.putText(
                annotated,
                f"D{i+1}",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated
    except Exception as e:
        print(f"[ERROR] annotate_defects: {e}")
        return original_image


def calculate_defect_metrics(defect_contours, image_shape):
    """
    Calculate defect statistics.

    Args:
        defect_contours: list of contours
        image_shape: tuple (height, width, channels) or (height, width)

    Returns:
        dict with keys: num_defects, total_defect_area, image_area, defect_percentage
    """
    try:
        num_defects = len(defect_contours)
        total_defect_area = sum(cv2.contourArea(c) for c in defect_contours)
        image_area = image_shape[0] * image_shape[1]
        defect_percentage = (total_defect_area / image_area) * 100 if image_area > 0 else 0.0

        return {
            "num_defects": num_defects,
            "total_defect_area": total_defect_area,
            "image_area": image_area,
            "defect_percentage": round(defect_percentage, 4),
        }
    except Exception as e:
        print(f"[ERROR] calculate_defect_metrics: {e}")
        return {
            "num_defects": 0,
            "total_defect_area": 0,
            "image_area": 0,
            "defect_percentage": 0.0,
        }


def classify_product(defect_percentage, pass_fail_threshold=2.0):
    """
    Classify a product as PASS or FAIL based on defect percentage.

    Args:
        defect_percentage: float, percentage of image covered by defects
        pass_fail_threshold: float, max allowed defect % to pass (default 2.0)

    Returns:
        tuple: (result: str, confidence: float)
            result is "PASS" or "FAIL"
            confidence is a 0–100 float
    """
    try:
        if defect_percentage <= pass_fail_threshold:
            result = "PASS"
            # Confidence increases as defect % is well below threshold
            margin = pass_fail_threshold - defect_percentage
            confidence = min(100.0, 50.0 + (margin / pass_fail_threshold) * 50.0)
        else:
            result = "FAIL"
            # Confidence increases as defect % is well above threshold
            excess = defect_percentage - pass_fail_threshold
            confidence = min(100.0, 50.0 + (excess / pass_fail_threshold) * 50.0)

        return result, round(confidence, 2)
    except Exception as e:
        print(f"[ERROR] classify_product: {e}")
        return "UNKNOWN", 0.0


def save_output_image(image, filename_prefix="result", output_dir="outputs"):
    """
    Save a processed image to the outputs directory with a timestamp.

    Args:
        image: numpy array (image to save)
        filename_prefix: str, prefix for the output filename
        output_dir: str, directory to save into

    Returns:
        str: full path of the saved file, or None on failure
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath
    except Exception as e:
        print(f"[ERROR] save_output_image: {e}")
        return None


def run_full_pipeline(image, canny_thresh1=50, canny_thresh2=150,
                      min_defect_area=100, pass_fail_threshold=2.0,
                      blur_kernel=5, save_outputs=False):
    """
    Execute the complete defect detection pipeline.

    Args:
        image: numpy array (BGR) - input image
        canny_thresh1: int, lower Canny threshold
        canny_thresh2: int, upper Canny threshold
        min_defect_area: int, minimum contour area to flag as defect
        pass_fail_threshold: float, max defect % allowed to pass
        blur_kernel: int, Gaussian blur kernel size
        save_outputs: bool, whether to save pipeline stages to disk

    Returns:
        dict containing all pipeline stages and results
    """
    results = {}

    # Stage 1: Grayscale conversion
    gray = convert_to_grayscale(image)
    results["grayscale"] = gray

    # Stage 2: Gaussian blur
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)
    results["blurred"] = blurred

    # Stage 3: Canny edge detection
    edges = apply_canny_edge_detection(blurred, canny_thresh1, canny_thresh2)
    results["edges"] = edges

    # Stage 4: Contour detection
    defect_contours = find_defect_contours(edges, min_defect_area)
    results["contours"] = defect_contours

    # Stage 5: Annotate original image
    annotated = annotate_defects(image, defect_contours)
    results["annotated"] = annotated

    # Stage 6: Compute metrics
    metrics = calculate_defect_metrics(defect_contours, image.shape)
    results.update(metrics)

    # Stage 7: Classify product
    verdict, confidence = classify_product(metrics["defect_percentage"], pass_fail_threshold)
    results["verdict"] = verdict
    results["confidence"] = confidence

    # Optional: save outputs
    if save_outputs:
        results["saved_annotated"] = save_output_image(annotated, "annotated")
        results["saved_edges"] = save_output_image(edges, "edges")

    return results
