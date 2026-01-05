from PIL import ImageDraw, ImageFont


COLORS = {
    "background": (0, 0, 255),
    "text": (255, 0, 0),
    "logo": (0, 200, 0),
    "graphic": (255, 165, 0),
}


def draw_debug(canvas, placements, output_path):
    """
    Draw bounding boxes and labels on rendered canvas
    """
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    for obj_type, bbox in placements:
        x1, y1, x2, y2 = bbox
        color = COLORS.get(obj_type, (255, 255, 255))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 3, y1 + 3), obj_type, fill=color, font=font)

    canvas.save(output_path)
