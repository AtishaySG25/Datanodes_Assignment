# from psd_tools import PSDImage
# from PIL import Image
# import os

# INPUT_PSD = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
# OUTPUT_DIR = "D:/Datanodes_Assignment/output_working_code"

# TARGET_SIZES = {
#     "970x90": (970, 90),
#     "728x90": (728, 90),
#     "160x600": (160, 600),
#     "468x60": (468, 60),
#     "200x200": (200, 200),
#     "300x250": (300, 250)
# }

# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def extract_layers(psd):
#     objects = []

#     for layer in psd:
#         if not layer.visible:
#             continue

#         img = layer.composite()
#         if img is None:
#             continue

#         # --- simple semantic classification ---
#         if layer.kind == "type":
#             obj_type, priority = "text", 3
#         elif layer.kind == "smartobject":
#             obj_type, priority = "logo", 2
#         elif layer.size == psd.size:
#             obj_type, priority = "background", 0
#         else:
#             obj_type, priority = "other", 1

#         objects.append((priority, obj_type, img))

#     # higher priority first
#     objects.sort(reverse=True, key=lambda x: x[0])
#     return objects


# def render_asset(objects, target_size, output_path):
#     W, H = target_size
#     canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

#     # --- background ---
#     for _, t, img in objects:
#         if t == "background":
#             bg = img.resize((W, H))
#             canvas.paste(bg, (0, 0))
#             break

#     # --- foreground placement ---
#     fg = [o for o in objects if o[1] != "background"]

#     if W > H:
#         # horizontal banners
#         x = 10
#         for _, _, img in fg:
#             scale = min((H - 20) / img.height, 1.0)
#             img = img.resize((int(img.width * scale), int(img.height * scale)))
#             y = (H - img.height) // 2
#             canvas.paste(img, (x, y), img)
#             x += img.width + 20
#             if x > W:
#                 break
#     else:
#         # vertical banner
#         y = 10
#         for _, _, img in fg:
#             scale = min((W - 20) / img.width, 1.0)
#             img = img.resize((int(img.width * scale), int(img.height * scale)))
#             x = (W - img.width) // 2
#             canvas.paste(img, (x, y), img)
#             y += img.height + 20
#             if y > H:
#                 break

#     canvas.save(output_path)


# def main():
#     psd = PSDImage.open(INPUT_PSD)
#     objects = extract_layers(psd)

#     for name, size in TARGET_SIZES.items():
#         out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
#         render_asset(objects, size, out_path)
#         print(f"Generated {out_path}")


# if __name__ == "__main__":
#     main()

from psd_tools import PSDImage
from PIL import Image
import os

INPUT_PSD = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
OUTPUT_DIR = "D:/Datanodes_Assignment/output_working_code_v2"

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "468x60": (468, 60),
    "200x200": (200, 200),
    "300x250": (300, 250)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def layout_type(W, H):
    r = W / H
    if r < 0.6:
        return "skyscraper"     # 160x600
    elif r < 1.2:
        return "box"            # 200x200, 300x250
    elif r < 4:
        return "banner"         # 468x60
    else:
        return "ultra_wide"     # 970x90, 728x90

def extract_layers(psd):
    objects = []

    for layer in psd:
        if not layer.visible:
            continue

        img = layer.composite()
        if img is None:
            continue

        # --- semantic classification (safe & simple) ---
        if layer.kind == "type":
            obj_type, priority = "text", 3
        elif layer.kind == "smartobject":
            obj_type, priority = "logo", 4
        elif layer.size == psd.size:
            obj_type, priority = "background", 0
        else:
            obj_type, priority = "other", 1

        objects.append({
            "priority": priority,
            "type": obj_type,
            "image": img.convert("RGBA")
        })

    # Higher priority first
    objects.sort(key=lambda x: x["priority"], reverse=True)
    return objects


def render_asset(objects, target_size, output_path):
    W, H = target_size
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    padding = int(0.04 * min(W, H))
    mode = layout_type(W, H)

    # ---- Background ----
    for obj in objects:
        if obj["type"] == "background":
            bg = obj["image"].resize((W, H), Image.LANCZOS)
            canvas.paste(bg, (0, 0))
            break

    fg = [o for o in objects if o["type"] != "background"]

    # =========================
    # 🟩 SKYSCRAPER (160x600)
    # =========================
    if mode == "skyscraper":
        y = padding
        for i, obj in enumerate(fg):
            img = obj["image"]
            scale = min(W / img.width * 0.9, 1.5)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            x = (W - img.width) // 2

            if y + img.height > H:
                break

            canvas.paste(img, (x, y), img)
            y += img.height + padding

    # =========================
    # 🟦 BOX (200x200 / 300x250)
    # =========================
    elif mode == "box":
        if not fg:
            canvas.save(output_path)
            return

        # Primary object fills most space
        main = fg[0]["image"]
        scale = min(W / main.width * 0.9, H / main.height * 0.9)
        main = main.resize((int(main.width * scale), int(main.height * scale)))
        canvas.paste(
            main,
            ((W - main.width) // 2, (H - main.height) // 2),
            main
        )

        # Secondary objects (corners)
        corners = [
            (padding, padding),
            (W - padding, padding),
            (padding, H - padding),
            (W - padding, H - padding),
        ]

        for obj, (cx, cy) in zip(fg[1:], corners):
            img = obj["image"]
            scale = min(W / img.width * 0.25, 1.0)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            x = cx - img.width if cx > W / 2 else cx
            y = cy - img.height if cy > H / 2 else cy
            canvas.paste(img, (x, y), img)

    # =========================
    # 🟨 BANNER (468x60)
    # =========================
    elif mode == "banner":
        # One dominant object centered
        main = fg[0]["image"]
        scale = min(H / main.height * 0.9, W / main.width * 0.9)
        main = main.resize((int(main.width * scale), int(main.height * scale)))
        canvas.paste(
            main,
            ((W - main.width) // 2, (H - main.height) // 2),
            main
        )

        # Optional logo on left
        if len(fg) > 1:
            logo = fg[1]["image"]
            scale = min(H / logo.height * 0.6, 1.0)
            logo = logo.resize((int(logo.width * scale), int(logo.height * scale)))
            canvas.paste(
                logo,
                (padding, (H - logo.height) // 2),
                logo
            )

    # =========================
    # 🟥 ULTRA-WIDE (existing)
    # =========================
    else:
        x = padding
        for obj in fg:
            img = obj["image"]
            scale = min((H - 2 * padding) / img.height, 1.0)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
            y = (H - img.height) // 2

            if x + img.width > W:
                break

            canvas.paste(img, (x, y), img)
            x += img.width + padding

    canvas.save(output_path, "PNG")



def main():
    psd = PSDImage.open(INPUT_PSD)
    objects = extract_layers(psd)

    for name, size in TARGET_SIZES.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        render_asset(objects, size, out_path)
        print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
