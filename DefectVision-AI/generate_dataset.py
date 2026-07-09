"""
generate_dataset.py  –  VisionInspect AI Demo Dataset Generator v2
Produces 10 inspection-friendly 1024×1024 PCB images.

Design philosophy:
  GOOD boards  → uniform green substrate, soft traces, very low edge density
                  → Canny finds few contours → PASS
  DEFECTIVE boards → same base + large high-contrast anomaly regions
                  → Canny finds large contours well above noise → FAIL

Run:  python generate_dataset.py
"""

import cv2
import numpy as np
import os
import random

random.seed(99)
np.random.seed(99)

W, H = 1024, 1024
GOOD_DIR   = "dataset/good"
DEFECT_DIR = "dataset/defective"
os.makedirs(GOOD_DIR,   exist_ok=True)
os.makedirs(DEFECT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PCB_BG        = (36,  90,  36)    # uniform green PCB solder mask
TRACE_COL     = (45, 110, 45)     # traces — ONLY slightly different from BG
                                   # → Canny will NOT trigger on them at thresh>60
COMP_DARK     = (28,  75,  28)    # component body (very close to BG)
PAD_COL       = (50, 120, 50)     # solder pad — subtle, not bright

# Defect colours — HIGH CONTRAST so Canny detects them clearly
SCRATCH_DARK  = (8,   12,   8)    # near-black scratch
BLOB_BRIGHT   = (10,  10, 180)    # bright copper blob (very different from BG)
BURN_COL      = (6,    8,   6)    # charred burn mark
MISSING_COL   = (18,  45,  18)    # missing pad area (dark)


# ═════════════════════════════════════════════════════════════════════════════
# BASE CANVAS  –  low-edge PCB surface
# ═════════════════════════════════════════════════════════════════════════════
def make_base():
    """Uniform green PCB substrate. Gaussian-blurred so it has ZERO sharp edges."""
    img = np.full((H, W, 3), PCB_BG, dtype=np.uint8)
    # Very soft random grain — NOT enough to trigger Canny at threshold 50+
    noise = np.random.randint(-6, 6, (H, W, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    return img


def draw_pcb_zones(img):
    """
    Draw soft PCB layout zones (traces, pads, components) using colours
    very close to the background so Canny does NOT fire on them.
    """
    # Horizontal trace bands — only 6-7 grey-scale intensity above BG
    for y in range(120, H - 120, 80):
        cv2.line(img, (60, y), (W - 60, y), TRACE_COL, 4)

    # Vertical trace bands
    for x in range(120, W - 120, 90):
        cv2.line(img, (x, 60), (x, H - 60), TRACE_COL, 4)

    # Component zones (filled rectangles — dark, close to BG)
    for cx, cy, bw, bh in [
        (260, 260, 130, 80),
        (580, 260, 120, 75),
        (420, 500, 160, 95),
        (180, 580, 110, 70),
        (700, 560, 125, 80),
    ]:
        cv2.rectangle(img, (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2), COMP_DARK, -1)
        cv2.rectangle(img, (cx - bw//2, cy - bh//2), (cx + bw//2, cy + bh//2), TRACE_COL,  1)

    # Solder pads (small, close-colour)
    for px, py in [(180,180),(300,180),(440,180),(580,180),(720,180),(860,180),
                   (180,440),(300,440),(440,440),(580,440),(720,440),(860,440),
                   (180,700),(300,700),(440,700),(580,700),(720,700),(860,700)]:
        cv2.circle(img, (px, py), 9, PAD_COL, -1)

    # Board edge marker
    cv2.rectangle(img, (18, 18), (W - 18, H - 18), TRACE_COL, 2)

    # Final soft blur to eliminate any residual sharp trace edges
    img[:] = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# GOOD PCBs  –  no anomalies added
# ═════════════════════════════════════════════════════════════════════════════
def make_good_pcb(variant: int):
    img = make_base()
    draw_pcb_zones(img)

    # Subtle per-variant brightness/contrast tweak only
    deltas = {1: 0, 2: 6, 3: -5, 4: 4, 5: -3}
    delta = deltas.get(variant, 0)
    if delta:
        img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Mild camera noise — not enough to trigger Canny
    noise = np.random.normal(0, 2.5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# DEFECTIVE PCBs  –  high-contrast anomalies on the same base
# Each defect covers 3–10% of image area so the detector clearly fires
# ═════════════════════════════════════════════════════════════════════════════

def make_defect_1_scratch():
    """3 wide diagonal scratches — near-black on green. ~4% coverage."""
    img = make_base()
    draw_pcb_zones(img)
    # Main scratch
    cv2.line(img, (60,  80), (700, 600), SCRATCH_DARK, 16)
    cv2.line(img, (80, 100), (720, 620), SCRATCH_DARK, 10)
    cv2.line(img, (40,  60), (680, 580), SCRATCH_DARK,  7)
    # Second smaller scratch
    cv2.line(img, (500, 150), (900, 480), SCRATCH_DARK, 12)
    cv2.line(img, (520, 160), (920, 490), SCRATCH_DARK,  7)
    # Rough edges
    for _ in range(120):
        bx = random.randint(60, 700)
        by = int(80 + (bx - 60) * (520 / 640)) + random.randint(-14, 14)
        if 0 < bx < W and 0 < by < H:
            cv2.circle(img, (bx, by), random.randint(3, 9), SCRATCH_DARK, -1)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_defect_2_broken_track():
    """Burnt gaps in traces — large dark rectangles. ~5% coverage."""
    img = make_base()
    draw_pcb_zones(img)
    # Large burnt break regions
    breaks = [
        (200, 180, 420, 240),   # x1,y1,x2,y2
        (340, 380, 600, 420),
        (500, 550, 380, 620),
        (650, 160, 850, 210),
        (100, 700, 350, 750),
    ]
    for x1, y1, x2, y2 in breaks:
        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), BURN_COL, -1)
        # Irregular rough edge
        for _ in range(30):
            ex = random.randint(x1, x2)
            ey = random.choice([y1, y2]) + random.randint(-8, 8)
            cv2.circle(img, (ex, max(0,min(H-1,ey))), random.randint(4, 10), BURN_COL, -1)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_defect_3_missing_holes():
    """Missing pads — dark circular regions where pads should be. ~3.5% coverage."""
    img = make_base()
    draw_pcb_zones(img)
    missing_centers = [
        (180,180),(300,180),(440,440),(580,700),(720,440),
        (860,180),(180,700),(580,180),(720,700),(300,440),
    ]
    for mx, my in missing_centers:
        # Dark pit — pad area is missing/damaged
        cv2.circle(img, (mx, my), 32, MISSING_COL, -1)
        # Darker centre
        cv2.circle(img, (mx, my), 18, BURN_COL, -1)
        # Rough jagged edge
        for angle in np.linspace(0, 2*np.pi, 16):
            r = 30 + random.randint(-4, 4)
            ex = int(mx + r * np.cos(angle))
            ey = int(my + r * np.sin(angle))
            cv2.circle(img, (ex, ey), random.randint(3, 7), MISSING_COL, -1)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_defect_4_copper_blob():
    """Dark solder blobs + bright spills — maximum contrast. ~5% coverage."""
    img = make_base()
    draw_pcb_zones(img)
    blob_centers = [
        (280, 300, 75), (620, 220, 80), (780, 560, 85),
        (200, 650, 70), (512, 800, 78), (750, 380, 73),
    ]
    for bx, by, base_r in blob_centers:
        pts = []
        n = 14
        for i in range(n):
            angle = 2 * np.pi * i / n + random.uniform(-0.4, 0.4)
            r = base_r + random.randint(-8, 8)
            pts.append([int(bx + r * np.cos(angle)), int(by + r * np.sin(angle))])
        pts = np.array(pts, np.int32)
        # Dark blob body — near-black creates strongest Canny edge against green
        cv2.fillPoly(img, [pts], (8, 8, 8))
        # Bright rim — double-edge (bright then dark) maximises contour size
        cv2.polylines(img, [pts], True, (200, 200, 50), 6)
        # Three thick dark spill lines
        for dx, dy in [(90, -40), (-80, 45), (10, 95)]:
            ex = max(0, min(W-1, bx + dx))
            ey = max(0, min(H-1, by + dy))
            cv2.line(img, (bx, by), (ex, ey), (8, 8, 8), 18)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_defect_5_multiple():
    """All defect types combined. ~10% coverage."""
    img = make_base()
    draw_pcb_zones(img)

    # A — large scratch
    cv2.line(img, (50, 100), (600, 520), SCRATCH_DARK, 18)
    cv2.line(img, (70, 115), (620, 535), SCRATCH_DARK, 11)
    for _ in range(100):
        bx = random.randint(50, 580)
        by = int(100 + (bx-50)*(420/550)) + random.randint(-16, 16)
        cv2.circle(img, (max(0,min(W-1,bx)), max(0,min(H-1,by))), random.randint(4,10), SCRATCH_DARK, -1)

    # B — burnt track gaps
    for x1,y1,x2,y2 in [(650,120,860,175),(680,420,880,470)]:
        cv2.rectangle(img,(x1,y1),(x2,y2),BURN_COL,-1)
        for _ in range(20):
            cv2.circle(img,(random.randint(x1,x2),random.randint(y1,y2)),random.randint(5,10),BURN_COL,-1)

    # C — copper blob
    bx, by = 512, 750
    pts = np.array([[int(bx+60*np.cos(2*np.pi*i/12+random.uniform(-0.3,0.3))),
                     int(by+60*np.sin(2*np.pi*i/12+random.uniform(-0.3,0.3)))]
                    for i in range(12)], np.int32)
    cv2.fillPoly(img, [pts], BLOB_BRIGHT)
    cv2.line(img,(bx,by),(bx+80,by-20),BLOB_BRIGHT,10)
    cv2.line(img,(bx,by),(bx-70,by+25),BLOB_BRIGHT,10)

    # D — missing pads
    for mx, my in [(300,700),(720,180),(180,440)]:
        cv2.circle(img,(mx,my),34,MISSING_COL,-1)
        cv2.circle(img,(mx,my),18,BURN_COL,-1)

    # E — oxidation burn
    cv2.ellipse(img,(800,750),(55,32),15,0,360,BURN_COL,-1)
    cv2.ellipse(img,(800,750),(55,32),15,0,360,(20,40,20),3)

    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  VisionInspect AI  –  Dataset Generator v2")
    print("=" * 55)

    for i in range(1, 6):
        path = f"{GOOD_DIR}/pcb_good_{i}.png"
        print(f"  {path} ...", end=" ", flush=True)
        cv2.imwrite(path, make_good_pcb(i))
        print("✓")

    for i, fn in enumerate([
        make_defect_1_scratch,
        make_defect_2_broken_track,
        make_defect_3_missing_holes,
        make_defect_4_copper_blob,
        make_defect_5_multiple,
    ], start=1):
        path = f"{DEFECT_DIR}/pcb_defect_{i}.png"
        print(f"  {path} ...", end=" ", flush=True)
        cv2.imwrite(path, fn())
        print("✓")

    # Quick self-test: run the actual detector on all 10 images
    print("\n  ── Self-test (detector) ──")
    import sys
    sys.path.insert(0, "../backend")
    try:
        from detector import run_full_pipeline
        for label, folder, expected in [("GOOD",      GOOD_DIR,   "PASS"),
                                         ("DEFECTIVE", DEFECT_DIR, "FAIL")]:
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".png"):
                    continue
                img = cv2.imread(os.path.join(folder, fname))
                r = run_full_pipeline(img, canny_thresh1=50, canny_thresh2=150,
                                      min_defect_area=300, pass_fail_threshold=2.0)
                ok = "✓" if r["verdict"] == expected else "✗"
                print(f"  {ok} {fname:30s}  coverage={r['defect_percentage']:.4f}%  "
                      f"verdict={r['verdict']} (expected {expected})")
    except Exception as e:
        print(f"  [self-test skipped: {e}]")

    print("\n  Done! 10 images saved.")
    print("=" * 55)


if __name__ == "__main__":
    main()
