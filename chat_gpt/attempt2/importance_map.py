def rank_objects(layers: list, classifications: list, saliency_data=None) -> list:
    class_map = {c["layer_id"]: c["category"] for c in classifications}

    weights = {
        "logo": 1.0,
        "text": 0.9,
        "product": 0.85,
        "cta": 0.8,
        "background": 0.1,
        "ignore": 0.05
    }

    max_x = max(l["bbox"][0] + l["bbox"][2] for l in layers)
    max_y = max(l["bbox"][1] + l["bbox"][3] for l in layers)
    canvas_area = max_x * max_y

    ranked = []

    for l in layers:
        x, y, w, h = l["bbox"]
        area_ratio = (w * h) / canvas_area
        z = l["z_index"]
        cat = class_map.get(l["id"], "ignore")

        center_bias = 1 - (abs(0.5 - (x + w / 2) / max_x) +
                           abs(0.5 - (y + h / 2) / max_y)) / 2

        importance = weights[cat] * (0.5 + area_ratio) * center_bias * (1 + z / len(layers))

        ranked.append({
            "layer_id": l["id"],
            "category": cat,
            "importance": round(importance, 3)
        })

    ranked.sort(key=lambda x: x["importance"], reverse=True)

    for i, r in enumerate(ranked):
        r["role"] = "primary" if i == 0 else "secondary" if i <= 2 else "tertiary"

    return ranked
