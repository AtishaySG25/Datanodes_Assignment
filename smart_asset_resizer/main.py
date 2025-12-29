import os
from src.psd_parser import extract_design_objects
from src.renderer import render

INPUT_PSD = "D:/Datanodes_Assignment/smart_asset_resizer/input/Axis_Multicap_fund.psd"
OUTPUT_DIR = "D:/Datanodes_Assignment/smart_asset_resizer/output/output3"

TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "468x60": (468, 60),
    "300x250": (300, 250),
    "200x200": (200, 200),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    objects = extract_design_objects(INPUT_PSD)
    print(f"Extracted {len(objects)} design objects")

    for name, size in TARGET_SIZES.items():
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        render(objects, size, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
