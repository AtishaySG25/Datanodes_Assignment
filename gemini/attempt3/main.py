import os
import argparse
from asset_transformer import (
    parse_psd_layers,
    detect_objects,
    smart_crop_background,
    reposition_objects,
    render_asset,
    ElementType
)

def main():
    # Setup Paths
    input_path = "D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd"
    output_dir = "D:/Datanodes_Assignment/gemini/attempt3/resized_assets"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Please place a .psd file there.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Define Target Sizes
    target_sizes = [
        (970, 90),   # Leaderboard
        (728, 90),   # Leaderboard
        (160, 600),  # Skyscraper
        (300, 250),  # Rect
        (1200, 300)  # Custom
    ]

    # 1. Parse PSD
    print(f"Processing {input_path}...")
    psd = parse_psd_layers(input_path)

    # 2. Object Detection & Classification
    elements = detect_objects(psd)
    
    # Separate Background from Foreground
    bg_element = next((e for e in elements if e.type == ElementType.BACKGROUND), None)
    
    # Fallback if no explicit background detected, use the largest element or create white
    if not bg_element:
        print("Warning: No background layer detected. Using white fill.")
        # Logic handled in render_asset defaults, or create a dummy element here
    
    # 3. Generate Variations
    for width, height in target_sizes:
        print(f"--- Generating {width}x{height} ---")
        
        # A. Smart Background Crop
        if bg_element:
            final_bg = smart_crop_background(bg_element, (width, height))
        else:
            final_bg = None

        # B. Reposition Foreground Objects
        positioned_elements = reposition_objects(elements, (width, height))
        
        # C. Render
        output_filename = os.path.join(output_dir, f"asset_{width}x{height}.jpg")
        render_asset(final_bg, positioned_elements, (width, height), output_filename)

    print("Processing Complete.")

if __name__ == "__main__":
    main()