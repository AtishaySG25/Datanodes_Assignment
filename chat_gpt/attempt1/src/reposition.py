import cv2
import numpy as np

def reposition_objects(original_image, objects, target_w, target_h):
    """
    Proper layout engine for extreme aspect ratios (e.g., 1200x300).
    Uses ORIGINAL image and bounding boxes.
    """
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Fill background with mean color
    bg_color = original_image.mean(axis=(0, 1)).astype(np.uint8)
    canvas[:] = bg_color

    if not objects:
        return canvas

    # Use the top-priority object only (usually subject + text cluster)
    obj = objects[0]
    x, y, w, h = obj["bbox"]

    extracted = original_image[y:y+h, x:x+w]

    # Scale to fit height safely
    scale = (target_h * 0.85) / h
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(extracted, (new_w, new_h))

    # Center vertically, align left
    x_offset = 40
    y_offset = (target_h - new_h) // 2

    # Clamp if too wide
    if new_w + x_offset > target_w:
        new_w = target_w - x_offset - 20
        resized = cv2.resize(extracted, (new_w, new_h))

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas
