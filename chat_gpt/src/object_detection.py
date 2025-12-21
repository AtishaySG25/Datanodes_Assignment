import cv2
import numpy as np

def detect_objects(image, saliency_map):
    """
    Detects text blocks, logos, CTAs using contours + saliency.
    """
    edges = cv2.Canny(image, 100, 200)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    h, w = image.shape[:2]

    for cnt in contours:
        aspect_ratio = bw / float(bh)
        # Heuristic: text blocks are wide and thin
        is_text_like = aspect_ratio > 2.5 and bh < 0.25 * h
        if is_text_like:
            weight *= 2.5   # HARD PRIORITY
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < 0.01 * w * h:
            continue

        saliency_score = saliency_map[y:y+bh, x:x+bw].mean() / 255.0
        centrality = 1 - (abs((x + bw/2) - w/2) / (w/2))

        weight = saliency_score * area * centrality

        objects.append({
            "bbox": (x, y, bw, bh),
            "weight": weight
        })

    return sorted(objects, key=lambda x: x["weight"], reverse=True)
