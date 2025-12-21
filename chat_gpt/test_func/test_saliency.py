import cv2
import numpy as np
from psd_tools import PSDImage

# -------- CONFIG --------
PSD_PATH = "D:/Datanodes_Assignment/chat_gpt/input/master_assets/4548556.psd"
OUTPUT_PATH = "D:/Datanodes_Assignment/chat_gpt/test_func/saliency_output.png"
# ------------------------

def compute_saliency(image):
    """
    Uses spectral residual saliency to detect visually prominent regions.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)

    saliency_map = (saliency_map * 255).astype("uint8")
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 0)

    return saliency_map

def load_psd_as_bgr(path):
    """
    Loads PSD, flattens layers, converts to OpenCV BGR format.
    """
    psd = PSDImage.open(path)
    img = psd.composite()

    if img is None:
        raise ValueError("Failed to composite PSD")

    img = np.array(img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def main():
    # Load PSD
    image = load_psd_as_bgr(PSD_PATH)
    print("Image loaded:", image.shape)

    # Compute saliency
    saliency_map = compute_saliency(image)
    print("Saliency computed:", saliency_map.shape)

    # Save result
    cv2.imwrite(OUTPUT_PATH, saliency_map)
    print(f"Saliency map saved to {OUTPUT_PATH}")

    # Optional: visualize
    cv2.imshow("Original", image)
    cv2.imshow("Saliency Map", saliency_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
