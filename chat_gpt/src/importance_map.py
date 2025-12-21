import numpy as np

def generate_importance_map(image_shape, objects):
    """
    Creates a weighted importance heatmap from detected objects.
    """
    importance_map = np.zeros(image_shape[:2], dtype=np.float32)

    for obj in objects:
        x, y, w, h = obj["bbox"]
        importance_map[y:y+h, x:x+w] += obj["weight"]

    importance_map /= (importance_map.max() + 1e-6)
    return importance_map
