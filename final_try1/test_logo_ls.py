import cv2
import numpy as np
from psd_tools import PSDImage
import os

# ---------------- CONFIG ----------------
INPUT_PSD = "D:/Datanodes_Assignment/lady.psd"
OUTPUT_IMAGE = "D:/Datanodes_Assignment/gpt_logo_landscape.png"

TARGET_W = 1200
TARGET_H = 300
# ----------------------------------------


def load_psd_as_bgr(path):
    """Load and flatten PSD into OpenCV BGR image."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    psd = PSDImage.open(path)
    composite = psd.composite()

    if composite is None:
        raise ValueError("Failed to composite PSD")

    img = np.array(composite.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detect_primary_region(image):
    """
    Detect the most visually important region (logo/text/subject cluster).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = image.shape[:2]
    best_score = -1
    best_bbox = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 0.02 * img_w * img_h:
            continue

        # Score: area + centrality
        centrality = 1 - abs((x + w / 2) - img_w / 2) / (img_w / 2)
        score = area * centrality

        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    return best_bbox


def create_logo_landscape(image):
    """
    Create a clean 1200x300 banner using object repositioning.
    """
    canvas = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

    # Background fill (mean color)
    bg_color = image.mean(axis=(0, 1)).astype(np.uint8)
    canvas[:] = bg_color

    bbox = detect_primary_region(image)
    if bbox is None:
        return canvas

    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    # Scale ROI to fit height
    scale = (TARGET_H * 0.85) / h
    new_w = int(w * scale)
    new_h = int(h * scale)

    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Position: left-aligned, vertically centered
    x_offset = 40
    y_offset = (TARGET_H - new_h) // 2

    # Clamp width if overflow
    if x_offset + new_w > TARGET_W:
        new_w = TARGET_W - x_offset - 20
        roi_resized = cv2.resize(roi, (new_w, new_h))

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = roi_resized
    return canvas


def main():
    image = load_psd_as_bgr(INPUT_PSD)
    banner = create_logo_landscape(image)

    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE, banner, [cv2.IMWRITE_PNG_COMPRESSION, 6])

    print("Logo Landscape created:", OUTPUT_IMAGE)
    print("Size:", banner.shape[1], "x", banner.shape[0])


if __name__ == "__main__":
    main()
