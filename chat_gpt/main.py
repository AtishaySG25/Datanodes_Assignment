import os
import cv2
from src.preprocess import load_master_asset
from src.saliency import compute_saliency
from src.object_detection import detect_objects
from src.importance_map import generate_importance_map
from src.smart_crop import smart_crop
from src.reposition import reposition_objects
from src.renderer import save_image

INPUT_PATH = "D:/Datanodes_Assignment/chat_gpt/input/master_assets/4548556.psd"
OUTPUT_DIR = "output/secondary_assets"

ASSETS = {
    "Image_Landscape": (1200, 628, "jpeg"),
    "Image_Square": (1200, 1200, "jpeg"),
    "Image_Portrait": (960, 1200, "jpeg"),
    "Logo_Landscape": (1200, 300, "png"),
    "Logo_Square": (1200, 1200, "png"),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = load_master_asset(INPUT_PATH)
saliency = compute_saliency(image)
objects = detect_objects(image, saliency)

importance_map = generate_importance_map(image.shape, objects)
for name, (w, h, fmt) in ASSETS.items():
    if h < 400:
        result = reposition_objects(image, objects, w, h)
    else:
        cropped = smart_crop(
            image,
            importance_map,
            objects,
            min(w, image.shape[1]),
            min(h, image.shape[0])
            )
        result = cv2.resize(cropped, (w, h))

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_master_{name}.{fmt}"
    )

    save_image(result, output_path, fmt)
    print(f"Saved: {output_path}")
