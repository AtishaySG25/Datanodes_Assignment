import cv2
import numpy as np
from psd_tools import PSDImage
from PIL import Image
import os

def load_master_asset(path):
    """
    Loads PSD or image and converts to OpenCV BGR format.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".psd":
        psd = PSDImage.open(path)
        img = psd.composite()
        img = np.array(img.convert("RGB"))
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
