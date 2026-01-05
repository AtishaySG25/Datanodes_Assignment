# detector.py (FINAL ROBUST VERSION – NO OPENCV CORE FUNCTIONS THAT MAY BE MISSING)
import numpy as np
import easyocr
from PIL import Image

# Initialize OCR reader once
reader = easyocr.Reader(['en'], gpu=False)

def detect_text_regions(image):
    """
    Returns list of bounding boxes [x, y, w, h] for detected text
    Uses EasyOCR on PIL Image directly
    """
    results = reader.readtext(np.array(image))
    boxes = []
    for (bbox, text, prob) in results:
        if prob < 0.5:  # Filter low-confidence detections
            continue
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x = int(min(x_coords))
        y = int(min(y_coords))
        w = int(max(x_coords) - x)
        h = int(max(y_coords) - y)
        boxes.append((x, y, w, h))
    return boxes

def get_saliency_map(image):
    """
    Pure NumPy + PIL synthetic saliency map – no OpenCV dependency at all.
    Strong center bias + edge boost using simple Sobel-like gradient.
    Works extremely well for typical centered ad designs.
    """
    # Convert to numpy array (H, W, C)
    arr = np.array(image)
    h, w = arr.shape[:2]

    # 1. Strong Gaussian center bias
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    sigma_x = w * 0.35
    sigma_y = h * 0.35
    gaussian = np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2)))

    # 2. Edge/contrast boost (simple gradient magnitude)
    # Convert to grayscale
    if arr.shape[2] == 4:  # RGBA
        gray = np.array(image.convert('L'))
    else:
        gray = np.array(image.convert('L'))

    # Sobel-like gradient (NumPy only)
    grad_x = np.diff(gray, axis=1, prepend=0)
    grad_y = np.diff(gray, axis=0, append=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

    # Dilate edges slightly to cover logos/text
    kernel = np.ones((15, 15), dtype=np.float32)
    dilated = np.zeros_like(gradient_magnitude)
    for i in range(h):
        for j in range(w):
            window = gradient_magnitude[max(0, i-7):i+8, max(0, j-7):j+8]
            if window.size > 0:
                pad_window = np.pad(window, ((0,15-window.shape[0]), (0,15-window.shape[1])), mode='constant')
                dilated[i, j] = np.max(pad_window * kernel)

    # Combine
    saliency_map = gaussian * 0.65 + dilated * 0.35
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    return saliency_map.astype(np.float32)

def get_key_regions(image):
    """
    Returns list of (x, y, w, h, weight) – key visual regions
    """
    h, w = image.height, image.width
    saliency_map = get_saliency_map(image)

    # Simple threshold + bounding boxes from high saliency areas
    threshold = 0.4
    high_saliency = saliency_map > threshold

    # Find connected regions manually (since no cv2.findContours)
    regions = []
    visited = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            if high_saliency[y, x] and not visited[y, x]:
                # Flood fill to find component
                component = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    component.append((cy, cx))
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and high_saliency[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(component) < 500:  # Filter tiny noise
                    continue
                ys = [p[0] for p in component]
                xs = [p[1] for p in component]
                regions.append((min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys), 2.0))

    # Add detected text regions with highest priority
    for box in detect_text_regions(image):
        x, y, bw, bh = box
        padding = max(40, int(min(bw, bh) * 0.3))
        x = max(0, x - padding)
        y = max(0, y - padding)
        bw = min(w - x, bw + 2 * padding)
        bh = min(h - y, bh + 2 * padding)
        regions.append((x, y, bw, bh, 6.0))  # Text is most important

    # Strong center fallback
    center_ratio = 0.75
    center_w = int(w * center_ratio)
    center_h = int(h * center_ratio)
    center_x = (w - center_w) // 2
    center_y = (h - center_h) // 2
    regions.append((center_x, center_y, center_w, center_h, 1.5))

    return regions