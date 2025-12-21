import cv2
import numpy as np

def smart_crop(image, importance_map, target_w, target_h):
    """
    Finds crop window that preserves maximum importance.
    """
    h, w = image.shape[:2]
    best_score = -1
    best_coords = (0, 0)

    step = 20

    for y in range(0, h - target_h + 1, step):
        for x in range(0, w - target_w + 1, step):
            score = importance_map[y:y+target_h, x:x+target_w].sum()
            if score > best_score:
                best_score = score
                best_coords = (x, y)

    x, y = best_coords
    return image[y:y+target_h, x:x+target_w]
