import logging
from psd_tools import PSDImage
from PIL import Image
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)

def parse_psd_layers(psd_path):
    """
    Parse PSD and extract background + foreground layers using robust heuristics.
    Returns:
        background_image (PIL.Image)
        foreground_objects (list of dicts)
    """
    psd = PSDImage.open(psd_path)
    canvas_w, canvas_h = psd.width, psd.height

    background = None
    foreground_objects = []

    # Iterate bottom → top (Photoshop stacking order)
    for layer in psd.descendants():
        if not layer.is_visible():
            continue
        if layer.bbox is None:
            continue

        image = layer.topil()
        x1, y1, x2, y2 = layer.bbox
        w = x2 - x1
        h = y2 - y1
        area_ratio = (w * h) / (canvas_w * canvas_h)

        name = (layer.name or "").lower()

        # ---- CLASSIFICATION HEURISTICS ----
        if layer.kind == "type":
            obj_type = "text"
        elif "logo" in name:
            obj_type = "logo"
        elif "cta" in name or "button" in name:
            obj_type = "cta"
        else:
            obj_type = "decorative"

        # ---- BACKGROUND DETECTION ----
        # If it covers ~90% of canvas and is NOT text → background
        if area_ratio > 0.9 and obj_type != "text":
            background = image
            logging.info("Background layer detected via area heuristic")
            continue

        # ---- FOREGROUND ----
        foreground_objects.append({
            "image": image,
            "bbox": (x1, y1, w, h),
            "type": obj_type,
            "area_ratio": area_ratio
        })

    if background is None:
        raise RuntimeError("Background layer could not be detected")

    logging.info(f"Foreground objects detected: {len(foreground_objects)}")
    return background, foreground_objects


def detect_objects(image):
    """
    Detect foreground objects using edge + contour analysis.
    Used mainly as fallback for flattened assets.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2GRAY)
    edges = cv2.Canny(gray, 120, 240)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    h, w = gray.shape

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        if area < 0.01 * w * h:
            continue

        detected.append({
            "bbox": (x, y, cw, ch),
            "area": area
        })

    return detected

def compute_saliency(image):
    """
    Compute visual saliency map using OpenCV Spectral Residual.
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(img)

    saliency_map = (saliency_map * 255).astype("uint8")
    return saliency_map

IMPORTANCE_WEIGHTS = {
    "logo": 5.0,
    "text": 4.0,
    "cta": 3.5,
    "decorative": 1.0
}

def build_importance_map(objects, saliency_map):
    """
    Combine object importance and saliency into a single importance heatmap.
    """
    importance = saliency_map.astype("float32")

    for obj in objects:
        x, y, w, h = obj["bbox"]
        weight = IMPORTANCE_WEIGHTS.get(obj["type"], 1.0)
        importance[y:y+h, x:x+w] += weight * 255

    importance = np.clip(importance, 0, 255)
    return importance.astype("uint8")

def smart_crop_background(bg_image, importance_map, target_size):
    """
    Crop background using importance map to preserve high-value regions.
    """
    th, tw = target_size
    h, w = importance_map.shape

    target_ratio = tw / th
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        projection = importance_map.sum(axis=0)
        center = projection.argmax()
        x1 = max(0, center - new_w // 2)
        x2 = min(w, x1 + new_w)
        cropped = bg_image.crop((x1, 0, x2, h))
    else:
        new_h = int(w / target_ratio)
        projection = importance_map.sum(axis=1)
        center = projection.argmax()
        y1 = max(0, center - new_h // 2)
        y2 = min(h, y1 + new_h)
        cropped = bg_image.crop((0, y1, w, y2))

    return cropped.resize((tw, th))

def reposition_objects(objects, target_size):
    """
    Reposition objects based on hierarchy and target aspect ratio.
    Fully defensive against invalid PSD layers.
    """
    tw, th = target_size
    is_wide = tw > th

    PRIORITY = {
        "logo": 0,
        "text": 1,
        "cta": 2,
        "decorative": 3
    }

    objects = sorted(objects, key=lambda o: PRIORITY.get(o["type"], 99))

    layout = []
    y_cursor = int(th * 0.15)

    for obj in objects:
        img = obj.get("image")

        # ---- DEFENSIVE GUARD ----
        if img is None:
            continue

        w, h = img.size
        if w == 0 or h == 0:
            continue

        # ---- SCALE PROPORTIONALLY ----
        scale = min(
            (tw * 0.35) / w if is_wide else (tw * 0.7) / w,
            (th * 0.4) / h,
            1.0
        )

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h))

        # ---- POSITIONING ----
        if is_wide:
            x = int(tw * 0.05)
        else:
            x = int((tw - new_w) / 2)

        y = y_cursor

        layout.append({
            "image": img,
            "position": (x, y)
        })

        y_cursor += new_h + int(th * 0.03)

        if y_cursor > th:
            break  # avoid overflow safely

    return layout


def render_asset(background, objects, target_size, output_path):
    """
    Render final asset with background and foreground objects.
    """
    canvas = background.copy().resize(target_size)

    for obj in objects:
        canvas.paste(obj["image"], obj["position"], obj["image"])

    canvas.save(output_path)