import numpy as np

def generate_importance_map(image_shape, objects):
    """
    Creates a weighted importance heatmap from detected objects.
    """
    padding = 20  # pixels
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_shape[1], x + w + padding)
    y2 = min(image_shape[0], y + h + padding)
    importance_map[y1:y2, x1:x2] += obj["weight"]
    
    for obj in objects:
        x, y, w, h = obj["bbox"]
        importance_map[y:y+h, x:x+w] += obj["weight"]

    importance_map /= (importance_map.max() + 1e-6)
    return importance_map
