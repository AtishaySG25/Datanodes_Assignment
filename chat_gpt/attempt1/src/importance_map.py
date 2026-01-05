import numpy as np

def generate_importance_map(image_shape, objects):
    """
    Creates a weighted importance heatmap from detected objects.
    Adds padding to protect text and important regions from cropping.
    """

    img_h, img_w = image_shape[:2]
    importance_map = np.zeros((img_h, img_w), dtype=np.float32)

    for obj in objects:
        x, y, w, h = obj["bbox"]
        weight = obj["weight"]
        padding = 40 if obj["weight"] > 0.7 else 15
        # ---- SAFE BOUNDS ----
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        importance_map[y1:y2, x1:x2] += weight

    # ---- NORMALIZATION ----
    max_val = importance_map.max()
    if max_val > 0:
        importance_map /= max_val

    return importance_map
