# cropper.py
import numpy as np
from PIL import Image

def smart_crop(image, target_size, key_regions):
    """
    Crops image to target aspect ratio while preserving key_regions
    Uses sliding window + scoring based on region coverage
    """
    tw, th = target_size
    iw, ih = image.size
    
    aspect_target = tw / th
    aspect_orig = iw / ih
    
    if abs(aspect_target - aspect_orig) < 0.05:
        # Almost same aspect → just resize
        return image.resize((tw, th), Image.LANCZOS)
    
    crops = []
    if aspect_target > aspect_orig:
        # Target is wider → crop height
        new_h = int(iw / aspect_target)
        for y in range(0, ih - new_h + 1, max(1, (ih - new_h) // 20)):
            score = 0
            for x, ry, rw, rh, weight in key_regions:
                overlap_h = max(0, min(y + new_h, ry + rh) - max(y, ry))
                if overlap_h > 0:
                    area = rw * overlap_h
                    score += area * weight
            crops.append((score, y))
    else:
        # Target is taller → crop width
        new_w = int(ih * aspect_target)
        for x in range(0, iw - new_w + 1, max(1, (iw - new_w) // 20)):
            score = 0
            for rx, y, rw, rh, weight in key_regions:
                overlap_w = max(0, min(x + new_w, rx + rw) - max(x, rx))
                if overlap_w > 0:
                    area = overlap_w * rh
                    score += area * weight
            crops.append((score, x))
    
    # Pick best crop
    crops.sort(reverse=True)
    best_offset = crops[0][1] if crops else 0
    
    if aspect_target > aspect_orig:
        cropped = image.crop((0, best_offset, iw, best_offset + new_h))
    else:
        new_w = int(ih * aspect_target)
        cropped = image.crop((best_offset, 0, best_offset + new_w, ih))
    
    return cropped.resize((tw, th), Image.LANCZOS)