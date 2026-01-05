import numpy as np

def smart_crop(image, importance_map, objects, target_w, target_h):
    """
    Smart cropping that:
    1. Maximizes total visual importance
    2. Enforces HARD constraints so high-priority objects (text) are never clipped
    """

    img_h, img_w = image.shape[:2]
    best_score = -1
    best_coords = None

    step = 20  # sliding window step (performance vs accuracy)

    for crop_y in range(0, img_h - target_h + 1, step):
        for crop_x in range(0, img_w - target_w + 1, step):

            # ---- HARD CONSTRAINT CHECK (Text Safety) ----
            valid_crop = True

            for obj in objects:
                x, y, w, h = obj["bbox"]

                # High-priority objects (text / logo)
                if obj["weight"] > 0.7:
                    fully_inside = (
                        crop_x <= x and
                        crop_y <= y and
                        crop_x + target_w >= x + w and
                        crop_y + target_h >= y + h
                    )

                    if not fully_inside:
                        valid_crop = False
                        break

            if not valid_crop:
                continue

            # ---- SOFT OPTIMIZATION (Importance Maximization) ----
            crop_importance = importance_map[
                crop_y:crop_y + target_h,
                crop_x:crop_x + target_w
            ].sum()

            if crop_importance > best_score:
                best_score = crop_importance
                best_coords = (crop_x, crop_y)

    # ---- FAILSAFE (if no valid crop found) ----
    if best_coords is None:
        # Center crop as absolute fallback
        center_x = (img_w - target_w) // 2
        center_y = (img_h - target_h) // 2
        best_coords = (max(center_x, 0), max(center_y, 0))

    x, y = best_coords
    return image[y:y + target_h, x:x + target_w]
