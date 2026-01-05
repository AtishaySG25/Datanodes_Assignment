import cv2
import numpy as np
from psd_tools import PSDImage
from PIL import Image


# ----------------------------------------------------
# 1. PSD PARSING
# ----------------------------------------------------
def parse_psd(psd_path):
    psd = PSDImage.open(psd_path, ignore_errors=True)

    # DO NOT touch layers (linked layers crash)
    flat_image = psd.topil().convert("RGB")

    return [], flat_image

# ----------------------------------------------------
# 2. SALIENCY MAP
# ----------------------------------------------------
def compute_saliency(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, sal_map = saliency.computeSaliency(img)

    return sal_map


# ----------------------------------------------------
# 3. OBJECT DETECTION (Contours from Saliency)
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
        if w * h > 800:  # remove noise
            boxes.append((x, y, w, h))

    return boxes


# ----------------------------------------------------
# 4. IMPORTANCE REGION (PSD + CV)
# ----------------------------------------------------
def get_importance_boxes(psd_layers, cv_boxes):
    important = []

    # Prefer text layers from PSD
    for layer in psd_layers:
        if layer["type"] == "text":
            b = layer["bbox"]
            important.append((b.x1, b.y1, b.width, b.height))

    # Fallback to CV detected boxes
    if not important:
        important.extend(cv_boxes)

    return important


# ----------------------------------------------------
# 5. SMART CROP (Aspect Ratio Aware)
# ----------------------------------------------------
def smart_crop(img, boxes, target_size):
    img_w, img_h = img.size
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    xs, ys, xe, ye = [], [], [], []
    for x, y, w, h in boxes:
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)

    cx = int((min(xs) + max(xe)) / 2)
    cy = int((min(ys) + max(ye)) / 2)

    crop_w = img_w
    crop_h = int(crop_w / target_ratio)

    if crop_h > img_h:
        crop_h = img_h
        crop_w = int(crop_h * target_ratio)

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)

    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)

    return img.crop((x1, y1, x2, y2))


# ----------------------------------------------------
# 6. RENDER FINAL BANNER
# ----------------------------------------------------
def render_banner(cropped_img, target_size, output_path):
    banner = cropped_img.resize(target_size, Image.LANCZOS)
    banner.save(output_path)
    print(f"Saved banner → {output_path}")


# ----------------------------------------------------
# 7. MAIN PIPELINE
# ----------------------------------------------------
def generate_banner(psd_path, output_path, target_size=(970, 90)):
    psd_layers, flat_img = parse_psd(psd_path)

    saliency = compute_saliency(flat_img)
    cv_boxes = detect_objects(saliency)

    important_boxes = get_importance_boxes(psd_layers, cv_boxes)

    cropped = smart_crop(flat_img, important_boxes, target_size)

    render_banner(cropped, target_size, output_path)


# ----------------------------------------------------
# 8. RUN
# ----------------------------------------------------
if __name__ == "__main__":
    INPUT_PSD = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
    OUTPUT_IMAGE = "D:/Datanodes_Assignment/single_dimensions/banner_970x90.png"

    generate_banner(INPUT_PSD, OUTPUT_IMAGE)
