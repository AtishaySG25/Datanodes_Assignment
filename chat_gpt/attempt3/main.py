import os
import logging
from ar_970_90 import *
#  from src.psd_parser import parse_psd_layers
# from src.saliency import compute_saliency
# from src.importance import build_importance_map
# from src.cropper import smart_crop_background
# from src.layout import reposition_objects
# from src.renderer import render_asset

logging.basicConfig(level=logging.INFO)

INPUT_PSD = "D:\\Datanodes_Assignment\\input\\Axis_Multicap_fund.psd"
OUTPUT_DIR = "D:\\Datanodes_Assignment\\chat_gpt\\attempt3\\output"

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "468x60": (468, 60)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    bg, objects = parse_psd_layers(INPUT_PSD)

    saliency = compute_saliency(bg)
    importance = build_importance_map(objects, saliency)

    for name, size in TARGET_SIZES.items():
        logging.info(f"Generating asset {name}")
        cropped_bg = smart_crop_background(bg, importance, size)
        layout = reposition_objects(objects, size)

        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        render_asset(cropped_bg, layout, size, out_path)

if __name__ == "__main__":
    main()
