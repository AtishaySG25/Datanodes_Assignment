import cv2
import numpy as np

def compute_saliency(image):
    """
    Uses spectral residual saliency to detect visually prominent regions.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)

    saliency_map = (saliency_map * 255).astype("uint8")
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 0)

    return saliency_map
