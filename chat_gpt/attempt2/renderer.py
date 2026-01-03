from PIL import Image


def render_canvas(target_size, layers, layout, classifications, output_path):
    tw, th = target_size
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    layout_map = {l["layer_id"]: l for l in layout}
    class_map = {c["layer_id"]: c["category"] for c in classifications}

    for layer in sorted(layers, key=lambda x: x["z_index"]):
        lid = layer["id"]
        img = layer["image"]
        if img is None or lid not in layout_map:
            continue

        if class_map.get(lid) == "background":
            bg = img.resize((tw, th), Image.LANCZOS)
            canvas.paste(bg, (0, 0), bg)
            continue

        info = layout_map[lid]
        scale = info["scale"]
        x, y = info["position"]

        w, h = img.size
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0:
            continue

        resized = img.resize((nw, nh), Image.LANCZOS)
        if x + nw > tw or y + nh > th:
            continue

        canvas.paste(resized, (x, y), resized)

    canvas.save(output_path, "PNG")
