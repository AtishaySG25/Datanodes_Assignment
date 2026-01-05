def compute_layout(canvas_size, target_size, layers, ranked_objects):
    tw, th = target_size
    aspect = tw / th

    if aspect > 3:
        mode = "ultra_wide"
    elif aspect < 0.7:
        mode = "portrait"
    else:
        mode = "box"

    layer_map = {l["id"]: l for l in layers}
    layout = []
    pad = int(0.05 * min(tw, th))

    cursor_x, cursor_y = pad, pad

    for obj in ranked_objects:
        l = layer_map[obj["layer_id"]]
        _, _, w, h = l["bbox"]

        if mode == "ultra_wide":
            scale = min(th / h * (0.8 if obj["role"] == "primary" else 0.6), 1.0)
            x = min(cursor_x, tw - int(w * scale) - pad)
            y = (th - int(h * scale)) // 2
            cursor_x += int(w * scale) + pad

        elif mode == "portrait":
            scale = min(tw / w * (0.9 if obj["role"] == "primary" else 0.7), 1.0)
            x = (tw - int(w * scale)) // 2
            y = cursor_y
            cursor_y += int(h * scale) + pad

        else:
            scale = min(tw / w * 0.8, th / h * 0.8, 1.0)
            x = (tw - int(w * scale)) // 2
            y = (th - int(h * scale)) // 2

        layout.append({
            "layer_id": l["id"],
            "scale": round(scale, 3),
            "position": (int(x), int(y)),
            "z_index": l["z_index"]
        })

    return layout
