from psd_tools import PSDImage
from PIL import Image


def parse_psd(psd_path: str) -> dict:
    psd = PSDImage.open(psd_path)

    canvas_w, canvas_h = psd.width, psd.height
    layers_data = []
    z_index = 0

    for layer in psd.descendants():
        if not layer.is_visible():
            continue

        if layer.kind == "adjustment":
            continue

        if not layer.bbox:
            continue

        x1, y1, x2, y2 = layer.bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue

        if layer.kind == "type":
            layer_type = "text"
        elif layer.kind == "smartobject":
            layer_type = "smart_object"
        elif layer.kind == "pixel":
            layer_type = "image"
        else:
            layer_type = "unknown"

        image = None
        try:
            pil = layer.topil()
            if pil:
                pil = pil.convert("RGBA")
                image = pil.crop((x1, y1, x2, y2))  # 🔥 FIX
        except Exception:
            image = None

        layers_data.append({
            "id": f"layer_{z_index}",
            "name": layer.name or f"layer_{z_index}",
            "type": layer_type,
            "bbox": (x1, y1, w, h),
            "z_index": z_index,
            "opacity": layer.opacity,
            "image": image
        })

        z_index += 1

    return {
        "canvas_size": (canvas_w, canvas_h),
        "layers": layers_data
    }
