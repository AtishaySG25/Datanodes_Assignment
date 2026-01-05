def classify_layers(layers: list, canvas_size: tuple) -> list:
    cw, ch = canvas_size
    canvas_area = cw * ch
    results = []

    for l in layers:
        x, y, w, h = l["bbox"]
        area_ratio = (w * h) / canvas_area
        t = l["type"]

        category = "ignore"
        confidence = 0.3

        if t == "text":
            category = "text"
            confidence = 0.95
            if h < 0.15 * ch and y > 0.6 * ch:
                category = "cta"
                confidence = 0.85

        elif area_ratio > 0.8:
            category = "background"
            confidence = 0.9

        elif area_ratio < 0.08 and t in ["image", "smart_object"] and y < 0.3 * ch:
            category = "logo"
            confidence = 0.75

        elif area_ratio >= 0.08 and t == "image":
            category = "product"
            confidence = 0.8

        results.append({
            "layer_id": l["id"],
            "category": category,
            "confidence": confidence
        })

    return results
