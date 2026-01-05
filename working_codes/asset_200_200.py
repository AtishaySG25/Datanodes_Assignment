import cv2
import numpy as np
from psd_tools import PSDImage
from PIL import Image


# ----------------------------------------------------
# 1. SAFE PSD PARSING (NO LAYER CRASHES)
# ----------------------------------------------------
def parse_psd(psd_path):
    psd = PSDImage.open(psd_path, ignore_errors=True)

    # We intentionally skip layer parsing to avoid linked-layer crashes
    flat_image = psd.topil().convert("RGB")

    return flat_image


# ----------------------------------------------------
# 2. SALIENCY (ROBUST FALLBACK)
# ----------------------------------------------------
def compute_saliency(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Use OpenCV saliency if available
    if hasattr(cv2, "saliency"):
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        _, sal_map = saliency.computeSaliency(img)
        return sal_map

    # Fallback: edge + contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    sal_map = cv2.normalize(edges.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)
    return sal_map


# ----------------------------------------------------
# 3. OBJECT DETECTION FROM SALIENCY
# ----------------------------------------------------
def detect_objects(saliency_map):
    thresh = (saliency_map * 255).astype("uint8")
    _, binary = cv2.threshold(thresh, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 600:  # noise filter
            boxes.append((x, y, w, h))

    return boxes


# ----------------------------------------------------
# 4. SMART CROP FOR SQUARE OUTPUT (200x200)
# ----------------------------------------------------
def smart_square_crop(img, boxes, target_size):
    img_w, img_h = img.size
    target_w, target_h = target_size
    crop_size = min(img_w, img_h)

    # If objects exist, center crop around them
    if boxes:
        xs, ys, xe, ye = [], [], [], []
        for x, y, w, h in boxes:
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        cx = int((min(xs) + max(xe)) / 2)
        cy = int((min(ys) + max(ye)) / 2)
    else:
        # Fallback: image center
        cx, cy = img_w // 2, img_h // 2

    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)

    x2 = min(img_w, x1 + crop_size)
    y2 = min(img_h, y1 + crop_size)

    cropped = img.crop((x1, y1, x2, y2))

    return cropped.resize(target_size, Image.LANCZOS)


# ----------------------------------------------------
# 5. MAIN PIPELINE
# ----------------------------------------------------
def generate_200x200(psd_path, output_path):
    TARGET_SIZE = (200, 200)

    flat_img = parse_psd(psd_path)
    saliency = compute_saliency(flat_img)
    boxes = detect_objects(saliency)

    final_img = smart_square_crop(flat_img, boxes, TARGET_SIZE)
    final_img.save(output_path)

    print(f"✅ Saved 200x200 asset → {output_path}")


# ----------------------------------------------------
# 6. RUN
# ----------------------------------------------------
if __name__ == "__main__":
    INPUT_PSD = "/content/Axis_Multicap_fund.psd"
    OUTPUT_IMAGE = "asset_200x200.png"

    generate_200x200(INPUT_PSD, OUTPUT_IMAGE)
