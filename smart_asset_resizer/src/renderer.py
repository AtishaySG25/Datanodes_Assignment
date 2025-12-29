from PIL import Image
from src.debug_draw import draw_debug

DEBUG = True
DEBUG_DIR = "D:/Datanodes_Assignment/smart_asset_resizer/output/output3/970x90.png"

def smart_crop_background(bg_img, target_size):
    target_w, target_h = target_size
    src_w, src_h = bg_img.size

    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        crop_box = (left, 0, left + new_w, src_h)
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        crop_box = (0, top, src_w, top + new_h)

    cropped = bg_img.crop(crop_box)
    return cropped.resize((target_w, target_h), Image.LANCZOS)

def scale_foreground(img, target_size, obj_type):
    target_w, target_h = target_size

    if obj_type == "text":
        max_ratio = 0.6
    elif obj_type == "logo":
        max_ratio = 0.5
    else:
        max_ratio = 0.4

    max_h = int(target_h * max_ratio)
    scale = min(max_h / img.height, 1.0)

    new_w = int(img.width * scale)
    new_h = int(img.height * scale)

    return img.resize((new_w, new_h), Image.LANCZOS)

def render(objects, target_size, output_path):
    canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))
    placements = []

    # ---- Background ----
    bg = next((o for o in objects if o.type == "background"), None)
    if bg:
        bg_img = smart_crop_background(bg.image, target_size)
        canvas.paste(bg_img, (0, 0))
        placements.append(("background", (0, 0, target_size[0], target_size[1])))

    foregrounds = [o for o in objects if o.type != "background"]

    cx, cy = target_size[0] // 2, target_size[1] // 2
    y_cursor = 20

    if target_size[0] > target_size[1]:
        # Horizontal
        x_cursor = 20
        for obj in foregrounds:
            img = scale_foreground(obj.image, target_size, obj.type)
            x, y = x_cursor, cy - img.height // 2
            canvas.paste(img, (x, y), img)

            placements.append((
                obj.type,
                (x, y, x + img.width, y + img.height)
            ))

            x_cursor += img.width + 20
            if x_cursor > target_size[0]:
                break
    else:
        # Vertical
        for obj in foregrounds:
            img = scale_foreground(obj.image, target_size, obj.type)
            x, y = cx - img.width // 2, y_cursor
            canvas.paste(img, (x, y), img)

            placements.append((
                obj.type,
                (x, y, x + img.width, y + img.height)
            ))

            y_cursor += img.height + 20
            if y_cursor > target_size[1]:
                break

    canvas.save(output_path)

    # ---- Debug output ----
    if DEBUG:
        import os
        os.makedirs(DEBUG_DIR, exist_ok=True)

        debug_path = os.path.join(
            DEBUG_DIR,
            "debug_" + output_path.split("/")[-1]
        )
        draw_debug(canvas.copy(), placements, debug_path)
