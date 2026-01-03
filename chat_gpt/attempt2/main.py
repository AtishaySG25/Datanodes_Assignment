from preprocess import parse_psd
from object_detection import classify_layers
from importance_map import rank_objects
from reposition import compute_layout
from renderer import render_canvas
import os

INPUT_PSD = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
OUTPUT_DIR = "D:/Datanodes_Assignment/chat_gpt/attempt2/output"

targets = [
    (970, 90),
    (728, 90),
    (160, 600),
    (468, 60),
    (200, 200),
    (300, 250)
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

scene = parse_psd(INPUT_PSD)
labels = classify_layers(scene["layers"], scene["canvas_size"])
ranked = rank_objects(scene["layers"], labels)

for w, h in targets:
    layout = compute_layout(
        scene["canvas_size"], (w, h), scene["layers"], ranked
    )

    render_canvas(
        (w, h),
        scene["layers"],
        layout,
        labels,
        f"{OUTPUT_DIR}/asset2_{w}_{h}.png"
    )
