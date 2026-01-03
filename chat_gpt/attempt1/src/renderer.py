import cv2
import os

def save_image(image, path, fmt):
    """
    Saves image with size constraints.
    """
    if fmt == "jpeg":
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
