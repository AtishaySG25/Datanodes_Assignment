import cv2
import numpy as np
from psd_tools import PSDImage
from PIL import Image


# ----------------------------------------------------
# 1. SAFE PSD LOADING (FLATTEN ONLY)
# ----------------------------------------------------
def load_psd_flat(psd_path):
    """
    Safely load PSD as a flattened image.
    Avoids crashes from linked layers / new PSD metadata.
    """
    psd = PSDImage.open(psd_path, ignore_errors=True)
    return psd.topil().convert("RGB")


# ----------------------------------------------------
# 2. SALIENCY MAP (ROBUST + FALLBACK)
# ----------------------------------------------------
def compute_saliency(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Preferred: OpenCV contrib saliency
    if hasattr(cv2, "saliency"):
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        _, sal_map = sal.computeSaliency(img)
        return sal_map

    # Fallback: edge + contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    return cv2.normalize(
        edges.astype("float32"), None, 0, 1, cv2.NORM_MINMAX
    )


# ----------------------------------------------------
# 3. DETECT IMPORTANT REGIONS
# ----------------------------------------------------
def detect_regions(saliency_map):
    thresh = (saliency_map * 255).astype("uint8")
    _, binary = cv2.threshold(thresh, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 800:  # moderate noise filter
            boxes.append((x, y, w, h))

    return boxes


# ----------------------------------------------------
# 4. SMART CROP FOR 300x250 (MEDIUM RECTANGLE)
# ----------------------------------------------------
def smart_rectangle_crop(img, boxes, target_size):
    img_w, img_h = img.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    # Determine importance center
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
        cx, cy = img_w // 2, img_h // 2

    # Decide crop orientation based on source aspect
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:
        # Source too wide → crop width
        crop_h = img_h
        crop_w = int(crop_h * target_ratio)
    else:
        # Source too tall → crop height
        crop_w = img_w
        crop_h = int(crop_w / target_ratio)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)

    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)

    cropped = img.crop((x1, y1, x2, y2))
    return cropped.resize(target_size, Image.LANCZOS)


# ----------------------------------------------------
# 5. MAIN PIPELINE
# ----------------------------------------------------
def generate_300x250(psd_path, output_path):
    TARGET_SIZE = (300, 250)

    flat_img = load_psd_flat(psd_path)
    saliency = compute_saliency(flat_img)
    boxes = detect_regions(saliency)

    final_img = smart_rectangle_crop(flat_img, boxes, TARGET_SIZE)
    final_img.save(output_path)

    print(f"✅ Saved 300x250 asset → {output_path}")


# ----------------------------------------------------
# 6. RUN
# ----------------------------------------------------
if __name__ == "__main__":
    INPUT_PSD = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
    OUTPUT_IMAGE = "D:/Datanodes_Assignment/output/secondary_assets"

    generate_300x250(INPUT_PSD, OUTPUT_IMAGE)
