from psd_tools import PSDImage
from PIL import Image
import os

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "468x60": (468, 60),
    "200x200": (200, 200),
    "300x250": (300, 250),
}

psd_path = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
output_dir = "D:/Datanodes_Assignment/output_new_script_v2"

def main(psd_path: str, output_dir: str):
    print("Loading PSD...")
    psd = load_psd(psd_path)
    objects = extract_psd_objects(psd)
    print("Starting asset generation...")
    for name, size in TARGET_SIZES.items():
        scene = build_virtual_psd(objects, size)
        image = flatten_virtual_psd(scene, size)

        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, f"{name}.png"))
        print(f"Generated {name}.png")

def load_psd(psd_path: str) -> PSDImage:
    """
    Load master PSD file.
    """
    return PSDImage.open(psd_path)

def extract_psd_objects(psd: PSDImage) -> list:
    """
    Extract composited PSD layers as semantic objects.

    Returns:
    [
        {
            "type": "background | logo | text | other",
            "priority": int,
            "image": PIL.Image,
        }
    ]
    """
    objects = []

    for layer in psd:
        if not layer.visible:
            continue

        img = layer.composite()
        if img is None:
            continue

        if layer.kind == "type":
            obj_type, priority = "text", 3
        elif layer.kind == "smartobject":
            obj_type, priority = "logo", 4
        elif layer.size == psd.size:
            obj_type, priority = "background", 0
        else:
            obj_type, priority = "other", 1

        objects.append({
            "type": obj_type,
            "priority": priority,
            "image": img.convert("RGBA"),
        })

    objects.sort(key=lambda o: o["priority"], reverse=True)
    return objects

def build_virtual_psd(objects: list, target_size: tuple) -> list:
    """
    Build a virtual PSD scene for a given target size.

    Returns:
    [
        {
            "image": PIL.Image,
            "position": (x, y),
            "z_index": int
        }
    ]
    """
    W, H = target_size
    layout = classify_layout(W, H)

    if layout == "skyscraper":
        return layout_skyscraper(objects, W, H)

    elif layout == "box":
        return layout_box(objects, W, H)

    elif layout == "banner":
        return layout_banner(objects, W, H)

    else:
        return layout_ultra_wide(objects, W, H)

def classify_layout(W: int, H: int) -> str:
    """
    Decide layout strategy based on aspect ratio.
    """
    r = W / H
    if r < 0.6:
        return "skyscraper"
    elif r < 1.2:
        return "box"
    elif r < 4:
        return "banner"
    else:
        return "ultra_wide"

def layout_skyscraper(objects, W, H):
    scene = []
    y = int(0.05 * H)

    for obj in objects:
        img = obj["image"]
        scale = min(W / img.width * 0.9, 1.5)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

        x = (W - img.width) // 2
        if y + img.height > H:
            break

        scene.append({
            "image": img,
            "position": (x, y),
            "z_index": obj["priority"],
        })
        y += img.height + 10

    return scene

def layout_box(objects, W, H):
    scene = []

    if not objects:
        return scene

    main = objects[0]["image"]
    scale = min(W / main.width * 0.9, H / main.height * 0.9)
    main = main.resize((int(main.width * scale), int(main.height * scale)))

    scene.append({
        "image": main,
        "position": ((W - main.width) // 2, (H - main.height) // 2),
        "z_index": objects[0]["priority"],
    })

    return scene

def layout_banner(objects, W, H):
    scene = []

    main = objects[0]["image"]
    scale = min(H / main.height * 0.9, W / main.width * 0.9)
    main = main.resize((int(main.width * scale), int(main.height * scale)))

    scene.append({
        "image": main,
        "position": ((W - main.width) // 2, (H - main.height) // 2),
        "z_index": objects[0]["priority"],
    })

    return scene

def layout_ultra_wide(objects, W, H):
    scene = []
    x = int(0.04 * W)

    for obj in objects:
        img = obj["image"]
        scale = min(H / img.height * 0.85, 1.0)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

        y = (H - img.height) // 2
        if x + img.width > W:
            break

        scene.append({
            "image": img,
            "position": (x, y),
            "z_index": obj["priority"],
        })

        x += img.width + 15

    return scene

def flatten_virtual_psd(scene: list, size: tuple) -> Image.Image:
    """
    Flatten virtual PSD scene into final image.
    """
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))

    for layer in sorted(scene, key=lambda l: l["z_index"]):
        canvas.paste(layer["image"], layer["position"], layer["image"])

    return canvas

if __name__ == "__main__":
    main(psd_path, output_dir)
