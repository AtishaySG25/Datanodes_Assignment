from psd_tools import PSDImage
from PIL import Image
import os

INPUT_PSD = "smart_asset_resizer/input/Axis_Multicap_fund.psd"
OUTPUT_DIR = "smart_asset_resizer/output/output2"

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_layers(psd):
    objects = []

    for layer in psd:
        if not layer.visible:
            continue

        img = layer.composite()
        if img is None:
            continue

        # --- simple semantic classification ---
        if layer.kind == "type":
            obj_type, priority = "text", 3
        elif layer.kind == "smartobject":
            obj_type, priority = "logo", 2
        elif layer.size == psd.size:
            obj_type, priority = "background", 0
        else:
            obj_type, priority = "other", 1

        objects.append((priority, obj_type, img))

    # higher priority first
    objects.sort(reverse=True, key=lambda x: x[0])
    return objects


def render_asset(objects, target_size, output_path):
    W, H = target_size
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # --- background ---
    for _, t, img in objects:
        if t == "background":
            bg = img.resize((W, H))
            canvas.paste(bg, (0, 0))
            break

    # --- foreground placement ---
    fg = [o for o in objects if o[1] != "background"]

    if W > H:
        # horizontal banners
        x = 10
        for _, _, img in fg:
            scale = min((H - 20) / img.height, 1.0)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            y = (H - img.height) // 2
            canvas.paste(img, (x, y), img)
            x += img.width + 20
            if x > W:
                break
    else:
        # vertical banner
        y = 10
        for _, _, img in fg:
            scale = min((W - 20) / img.width, 1.0)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            x = (W - img.width) // 2
            canvas.paste(img, (x, y), img)
            y += img.height + 20
            if y > H:
                break

    canvas.save(output_path)


def main():
    psd = PSDImage.open(INPUT_PSD)
    objects = extract_layers(psd)

    for name, size in TARGET_SIZES.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        render_asset(objects, size, out_path)
        print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
