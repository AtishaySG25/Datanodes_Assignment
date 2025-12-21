import cv2
import numpy as np

def reposition_objects(image, objects, target_w, target_h):
    """
    Rearranges key elements for extreme aspect ratios (e.g., 1200x300).
    """
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # background fill
    bg_color = image.mean(axis=(0, 1)).astype(np.uint8)
    canvas[:] = bg_color

    x_cursor = 20
    y_center = target_h // 2

    for obj in objects[:3]:
        x, y, w, h = obj["bbox"]
        obj_img = image[y:y+h, x:x+w]

        scale = min((target_h * 0.8) / h, (target_w * 0.3) / w)
        new_w, new_h = int(w * scale), int(h * scale)

        obj_img = cv2.resize(obj_img, (new_w, new_h))
        y_pos = y_center - new_h // 2

        if x_cursor + new_w < target_w:
            canvas[y_pos:y_pos+new_h, x_cursor:x_cursor+new_w] = obj_img
            x_cursor += new_w + 30

    return canvas
