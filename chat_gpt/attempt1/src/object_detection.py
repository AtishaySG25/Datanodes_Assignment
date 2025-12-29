import cv2
import numpy as np

def detect_objects(image, saliency_map):
    """
    Detects text blocks, logos, and CTAs using contours + saliency.
    Text-like regions are given HARD priority.
    """

    edges = cv2.Canny(image, 100, 200)
    edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []
    img_h, img_w = image.shape[:2]

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        # ---- FILTER SMALL NOISE ----
        if area < 0.01 * img_w * img_h:
            continue

        # ---- SALIENCY SCORE ----
        saliency_score = saliency_map[y:y+bh, x:x+bw].mean() / 255.0

        # ---- CENTRALITY SCORE ----
        centrality = 1 - abs((x + bw / 2) - img_w / 2) / (img_w / 2)

        # ---- BASE WEIGHT ----
        weight = saliency_score * area * centrality

        # ---- TEXT HEURISTIC ----
        aspect_ratio = bw / float(bh + 1e-6)
        is_text_like = aspect_ratio > 2.5 and bh < 0.25 * img_h

        if is_text_like:
            weight *= 2.5  # HARD PRIORITY FOR TEXT

        objects.append({
            "bbox": (x, y, bw, bh),
            "weight": weight
        })

    return sorted(objects, key=lambda x: x["weight"], reverse=True)
