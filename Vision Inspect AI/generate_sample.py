"""
generate_sample.py
Generates synthetic industrial product test images in the dataset/ folder.
Run once: python generate_sample.py
"""

import cv2
import numpy as np
import os

os.makedirs("dataset", exist_ok=True)


def make_pcb_sample(path):
    """Create a synthetic PCB-like surface with scratch defects."""
    h, w = 480, 640
    img = np.full((h, w, 3), (30, 60, 30), dtype=np.uint8)  # dark green board

    # Draw circuit traces
    for y in range(40, h, 60):
        cv2.line(img, (20, y), (w - 20, y), (0, 180, 0), 2)
    for x in range(40, w, 80):
        cv2.line(img, (x, 20), (x, h - 20), (0, 160, 0), 2)

    # Draw solder pads
    for cx in range(80, w - 60, 80):
        for cy in range(80, h - 60, 60):
            cv2.circle(img, (cx, cy), 10, (0, 220, 180), -1)

    # Add defects (scratches / missing pads)
    cv2.line(img, (100, 80), (220, 160), (0, 0, 0), 5)    # scratch 1
    cv2.line(img, (350, 200), (410, 320), (0, 0, 0), 4)   # scratch 2
    cv2.ellipse(img, (500, 350), (25, 12), 30, 0, 360, (0, 0, 0), -1)  # burn mark

    # Add Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img)
    print(f"Saved: {path}")


def make_metal_sample(path):
    """Create a synthetic metal surface with cracks and dents."""
    h, w = 480, 640
    # Metallic grey base with slight texture
    base = np.random.randint(140, 175, (h, w), dtype=np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Simulate surface grain
    grain = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Add defects
    # Crack 1
    pts = np.array([[80, 100], [130, 140], [160, 200], [140, 260]], np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 40), 3)

    # Crack 2
    pts2 = np.array([[400, 80], [420, 130], [460, 170], [500, 210]], np.int32)
    cv2.polylines(img, [pts2], False, (40, 40, 40), 3)

    # Dent (dark ellipse)
    cv2.ellipse(img, (300, 300), (40, 20), 15, 0, 360, (60, 60, 60), -1)
    cv2.ellipse(img, (300, 300), (38, 18), 15, 0, 360, (80, 80, 80), 2)

    cv2.imwrite(path, img)
    print(f"Saved: {path}")


def make_clean_sample(path):
    """Create a clean product with no significant defects (should PASS)."""
    h, w = 480, 640
    base = np.random.randint(180, 210, (h, w), dtype=np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Very fine surface texture only
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(path, img)
    print(f"Saved: {path}")


if __name__ == "__main__":
    make_pcb_sample("dataset/sample_pcb_defective.png")
    make_metal_sample("dataset/sample_metal_defective.png")
    make_clean_sample("dataset/sample_clean_pass.png")
    print("\nDone! Three test images created in dataset/")
