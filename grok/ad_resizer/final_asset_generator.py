# smart_asset_resizer/main.py

from psd_tools import PSDImage
from PIL import Image
import os

# ==================== CONFIG ====================
INPUT_PSD = "D:/Datanodes_Assignment/grok/ad_resizer/input/Axis_Multicap_fund.psd"
OUTPUT_DIR = "D:/Datanodes_Assignment/grok/ad_resizer/output/output2"

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "468x60": (468, 60),
    "300x250": (300, 250),
    "200x200": (200, 200),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Safe margins (in pixels) — prevents elements from touching edges
MARGIN = 15


def extract_layers(psd):
    """
    Extract visible layers and classify them by importance
    Returns list of (priority, type, image, original_size)
    """
    elements = []

    for layer in psd:
        if not layer.is_visible():
            continue

        # Composite the layer (handles effects, blends, text rendering perfectly)
        img = layer.composite()
        if img is None or img.width == 0 or img.height == 0:
            continue

        # Classify layer
        if layer.is_group():
            layer_type = "group"
            priority = 1
        elif layer.kind == "type":
            layer_type = "text"
            priority = 10        # Highest priority
        elif layer.kind == "smartobject":
            layer_type = "logo"
            priority = 8
        elif layer.size == psd.size:  # Full canvas size → likely background
            layer_type = "background"
            priority = 0
        else:
            layer_type = "graphic"
            priority = 5

        elements.append((priority, layer_type, img.convert("RGBA"), img.size))

    # Sort by priority descending (most important first)
    elements.sort(reverse=True, key=lambda x: x[0])
    return elements


def render_asset(elements, target_size, output_path):
    """
    Render a new asset by intelligently placing layers on target canvas
    """
    target_w, target_h = target_size
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))  # Transparent initially

    # Separate background and foreground
    backgrounds = [e for e in elements if e[1] == "background"]
    foregrounds = [e for e in elements if e[1] != "background"]

    # === 1. Paste and resize background ===
    if backgrounds:
        bg_img = backgrounds[0][2]  # Take highest priority background
        bg_resized = bg_img.resize((target_w, target_h), Image.LANCZOS)
        canvas.paste(bg_resized, (0, 0))
    else:
        # Fallback: white background
        canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))

    # === 2. Place foreground elements smartly ===
    available_width = target_w - 2 * MARGIN
    available_height = target_h - 2 * MARGIN

    # Filter only important elements (text + logo + key graphics)
    key_elements = [e for e in foregrounds if e[1] in ("text", "logo", "graphic")]

    if not key_elements:
        canvas.save(output_path)
        return

    # For narrow horizontal banners: stack horizontally with centering
    if target_w > target_h:  # Landscape (e.g., 970x90)
        total_width = 0
        scaled_elements = []

        for _, elem_type, img, orig_size in key_elements:
            # Scale to fit height safely
            scale = min(available_height / img.height, 1.0)
            new_h = int(img.height * scale)
            new_w = int(img.width * scale)
            scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
            scaled_elements.append((scaled_img, elem_type))
            total_width += new_w

        # Add spacing
        num_gaps = len(scaled_elements) - 1
        spacing = 30 if num_gaps > 0 else 0
        if total_width + spacing * num_gaps > available_width and len(scaled_elements) > 1:
            # Too wide → reduce spacing or drop lower priority later if needed
            spacing = max(10, (available_width - total_width) // num_gaps)

        # Center the entire group horizontally
        current_x = (target_w - (total_width + spacing * num_gaps)) // 2

        for scaled_img, _ in scaled_elements:
            y = MARGIN + (available_height - scaled_img.height) // 2
            canvas.paste(scaled_img, (current_x, y), scaled_img)
            current_x += scaled_img.width + spacing

    else:  # Portrait or square (e.g., 160x600, 300x250)
        total_height = 0
        scaled_elements = []

        for _, elem_type, img, orig_size in key_elements:
            scale = min(available_width / img.width, 1.0)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
            scaled_elements.append((scaled_img, elem_type))
            total_height += new_h

        num_gaps = len(scaled_elements) - 1
        spacing = 25 if num_gaps > 0 else 0

        current_y = (target_h - (total_height + spacing * num_gaps)) // 2

        for scaled_img, _ in scaled_elements:
            x = MARGIN + (available_width - scaled_img.width) // 2
            canvas.paste(scaled_img, (x, current_y), scaled_img)
            current_y += scaled_img.height + spacing

    # Save as PNG to preserve quality and transparency
    canvas.save(output_path, optimize=True)
    print(f"Saved: {output_path}")


def main():
    print(f"Loading PSD: {INPUT_PSD}")
    psd = PSDImage.open(INPUT_PSD)
    print(f"PSD Size: {psd.size}, Layers: {len(psd)}")

    elements = extract_layers(psd)
    print(f"Extracted {len(elements)} visible elements")

    for name, size in TARGET_SIZES.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        render_asset(elements, size, out_path)


if __name__ == "__main__":
    main()