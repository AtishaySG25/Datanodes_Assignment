# main.py
import os
from psd_loader import load_psd_as_image
from detector import get_key_regions
from cropper import smart_crop

# Target sizes as requested
TARGET_SIZES = {
    "970x90": (970, 90),
    "728x90": (728, 90),
    "160x600": (160, 600),
    "468x60": (468, 60),
    "200x200": (200, 200),
    "300x250": (300, 250),
}

def process_master_psd(psd_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading PSD: {psd_path}")
    image = load_psd_as_image(psd_path)
    print(f"Master size: {image.size}")
    
    print("Detecting key visual elements (text + saliency)...")
    key_regions = get_key_regions(image)
    print(f"Found {len(key_regions)} important regions")
    
    for name, size in TARGET_SIZES.items():
        print(f"Generating {name}...")
        cropped = smart_crop(image, size, key_regions)
        output_path = os.path.join(output_dir, f"{name}.jpg")
        # Convert to RGB for JPEG
        rgb_image = cropped.convert('RGB')
        rgb_image.save(output_path, quality=95, optimize=True)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    psd_file = "D:/Datanodes_Assignment/grok/ad_resizer/input/Axis_Multicap_fund.psd"
    process_master_psd(psd_file)