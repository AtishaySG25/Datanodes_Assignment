from psd_tools import PSDImage
from PIL import Image
import os

# ==================== CONFIG ====================
INPUT_PSD = "D:/Datanodes_Assignment/smart_asset_resizer/input/Axis_Multicap_fund.psd"
OUTPUT_DIR = "D:/Datanodes_Assignment/grok/ad_resizer/output/output3"

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "468x60": (468, 60),
    "300x250": (300, 250),
    "200x200": (200, 200),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Safe margins (in pixels)
MARGIN = 15


def extract_layers(psd):
    """
    Extract visible layers and classify them by importance
    Returns list of (priority, type, image)
    """
    elements = []

    for layer in psd:
        # ❌ layer.is_visible() does NOT exist in psd-tools
        if not layer.visible:
            continue

        # Skip groups — groups cannot be composited directly
        if layer.is_group():
            continue

        img = layer.composite()
        if img is None or img.width == 0 or img.height == 0:
            continue

        # ---- Classification ----
        if layer.kind == "type":
            layer_type = "text"
            priority = 10
        elif layer.kind == "smartobject":
            layer_type = "logo"
            priority = 8
        elif layer.size == psd.size:
            layer_type = "background"
            priority = 0
        else:
            layer_type = "graphic"
            priority = 5

        elements.append((priority, layer_type, img.convert("RGBA")))

    # Highest priority first
    elements.sort(key=lambda x: x[0], reverse=True)
    return elements


def render_asset(elements, target_size, output_path):
    target_w, target_h = target_size
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))

    # Separate background and foreground
    backgrounds = [e for e in elements if e[1] == "background"]
    foregrounds = [e for e in elements if e[1] != "background"]

    # ---- Background ----
    if backgrounds:
        bg_img = backgrounds[0][2]
        bg_resized = bg_img.resize((target_w, target_h), Image.LANCZOS)
        canvas.paste(bg_resized, (0, 0))

    available_w = target_w - 2 * MARGIN
    available_h = target_h - 2 * MARGIN

    if not foregrounds:
        canvas.save(output_path)
        return

    # ---- Horizontal banners ----
    if target_w > target_h:
        x = MARGIN
        y_center = target_h // 2

        for _, _, img in foregrounds:
            scale = min(available_h / img.height, 1.0)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)

            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            y = y_center - new_h // 2
            if x + new_w > target_w - MARGIN:
                break

            canvas.paste(img_resized, (x, y), img_resized)
            x += new_w + 20

    # ---- Vertical / square banners ----
    else:
        y = MARGIN
        x_center = target_w // 2

        for _, _, img in foregrounds:
            scale = min(available_w / img.width, 1.0)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)

            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            x = x_center - new_w // 2
            if y + new_h > target_h - MARGIN:
                break

            canvas.paste(img_resized, (x, y), img_resized)
            y += new_h + 20

    canvas.save(output_path, optimize=True)
    print(f"Saved: {output_path}")


def main():
    print(f"Loading PSD: {INPUT_PSD}")
    psd = PSDImage.open(INPUT_PSD)
    print(f"PSD Size: {psd.size}, Layers: {len(psd)}")

    elements = extract_layers(psd)
    print(f"Extracted {len(elements)} elements")

    for name, size in TARGET_SIZES.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        render_asset(elements, size, out_path)


if __name__ == "__main__":
    main()
