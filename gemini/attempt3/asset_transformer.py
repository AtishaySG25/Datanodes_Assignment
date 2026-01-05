import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
import logging
import os
from enum import Enum

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ElementType(Enum):
    LOGO = "logo"
    TEXT = "text"
    CTA = "cta"
    DECORATIVE = "decorative"
    BACKGROUND = "background"
    UNKNOWN = "unknown"

class GraphicElement:
    def __init__(self, image: Image.Image, name: str, bbox: tuple):
        self.original_image = image
        self.image = image
        self.name = name
        self.original_bbox = bbox  # (x1, y1, x2, y2)
        self.type = ElementType.UNKNOWN
        self.saliency_score = 0.0
        self.area = image.width * image.height
        
    def to_cv2(self):
        """Convert PIL image to OpenCV format (BGR)."""
        open_cv_image = np.array(self.image)
        # Convert RGB to BGR
        return open_cv_image[:, :, ::-1].copy()

def compute_saliency(image: np.ndarray) -> np.ndarray:
    """
    Computes the saliency map using Spectral Residual.
    """
    try:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, map_ = saliency.computeSaliency(image)
        if success:
            # Normalize to 0-255
            map_ = (map_ * 255).astype("uint8")
            return map_
    except Exception as e:
        logger.warning(f"Saliency computation failed: {e}. Fallback to edge density.")
    
    # Fallback: Edge detection as proxy for saliency
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)

def classify_element(element: GraphicElement, saliency_map: np.ndarray) -> ElementType:
    """
    Classifies an element based on heuristics, layer names, and CV properties.
    """
    name_lower = element.name.lower()
    
    # 1. Metadata Heuristics
    if 'bg' in name_lower or 'background' in name_lower:
        return ElementType.BACKGROUND
    if 'logo' in name_lower:
        return ElementType.LOGO
    if 'cta' in name_lower or 'button' in name_lower:
        return ElementType.CTA
    if 'text' in name_lower or 'headline' in name_lower or 'copy' in name_lower:
        return ElementType.TEXT
        
    # 2. Computer Vision Heuristics
    w, h = element.image.size
    aspect = w / h if h > 0 else 0
    
    if aspect > 4.0:
        return ElementType.TEXT
    
    if w < 100 and h < 100:
        return ElementType.DECORATIVE
        
    return ElementType.UNKNOWN

def detect_objects(psd: PSDImage) -> list[GraphicElement]:
    """
    Extracts layers, converts to CV objects, detects content type, and scores importance.
    """
    logger.info("Detecting objects from PSD layers...")
    elements = []
    
    # Render composite for global context saliency
    composite = psd.composite()
    composite_cv = np.array(composite)[:, :, ::-1] # RGB to BGR
    global_saliency = compute_saliency(composite_cv)
    
    for layer in psd:
        if not layer.is_visible():
            continue
            
        # Get layer image
        layer_img = layer.composite()
        if layer_img is None:
            continue
            
        # FIXED: Handle bbox tuple correctly
        bbox = layer.bbox # Returns (left, top, right, bottom)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width == 0 or height == 0:
            continue
            
        # Create Element Wrapper
        element = GraphicElement(layer_img, layer.name, bbox)
        
        # Calculate local saliency score based on global map intersection
        x1, y1, x2, y2 = bbox
        
        # Ensure bounds are within image dimensions to avoid indexing errors
        h_map, w_map = global_saliency.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_map, x2), min(h_map, y2)
        
        if x2 > x1 and y2 > y1:
            region_saliency = global_saliency[y1:y2, x1:x2]
            element.saliency_score = np.mean(region_saliency)
        
        # Classify
        element.type = classify_element(element, global_saliency)
        
        # Filter out full background layers from the object list (handle separately)
        if element.type == ElementType.BACKGROUND or (width >= psd.width and height >= psd.height):
            logger.info(f"Identified Background Layer: {layer.name}")
            elements.insert(0, element) # Keep at start
        else:
            elements.append(element)
            
    logger.info(f"Detected {len(elements)} relevant elements.")
    return elements

def smart_crop_background(bg_element: GraphicElement, target_size: tuple) -> Image.Image:
    """
    Resizes and crops background to fill target canvas without black bars.
    """
    target_w, target_h = target_size
    bg_img = bg_element.original_image
    bg_w, bg_h = bg_img.size
    
    # Calculate scale needed to cover the target
    scale_w = target_w / bg_w
    scale_h = target_h / bg_h
    scale = max(scale_w, scale_h) # 'Cover' strategy
    
    new_w = int(bg_w * scale)
    new_h = int(bg_h * scale)
    
    resized_bg = bg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center Crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    
    cropped_bg = resized_bg.crop((left, top, right, bottom))
    return cropped_bg

def reposition_objects(elements: list[GraphicElement], target_size: tuple) -> list[tuple]:
    """
    Calculates new (x, y) coordinates and scales for objects based on visual hierarchy.
    """
    target_w, target_h = target_size
    aspect_ratio = target_w / target_h
    
    # Filter out background
    foreground = [e for e in elements if e.type != ElementType.BACKGROUND]
    
    # Sort by Hierarchy: Logo -> Text -> CTA -> Decor
    hierarchy_order = {
        ElementType.LOGO: 1,
        ElementType.TEXT: 2,
        ElementType.CTA: 3,
        ElementType.DECORATIVE: 4,
        ElementType.UNKNOWN: 5
    }
    foreground.sort(key=lambda x: hierarchy_order.get(x.type, 5))
    
    processed_elements = [] # List of (Image, x, y)
    
    # PADDING CONSTANTS
    padding_x = int(target_w * 0.05)
    padding_y = int(target_h * 0.05)
    
    if aspect_ratio > 2.0:
        # LANDSCAPE / LEADERBOARD
        logger.info("Layout Strategy: Horizontal Flow")
        current_x = padding_x
        
        for el in foreground:
            max_h = int(target_h * 0.8)
            ratio = max_h / el.image.height if el.image.height > 0 else 1
            new_w = int(el.image.width * ratio)
            new_h = max_h
            
            if new_w > target_w * 0.4:
                ratio = (target_w * 0.4) / el.image.width if el.image.width > 0 else 1
                new_w = int(el.image.width * ratio)
                new_h = int(el.image.height * ratio)

            resized_img = el.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            y_pos = (target_h - new_h) // 2
            
            processed_elements.append((resized_img, current_x, y_pos))
            current_x += new_w + padding_x
            
    elif aspect_ratio < 0.5:
        # SKYSCRAPER
        logger.info("Layout Strategy: Vertical Flow")
        current_y = padding_y
        
        for el in foreground:
            max_w = int(target_w * 0.8)
            ratio = max_w / el.image.width if el.image.width > 0 else 1
            new_w = max_w
            new_h = int(el.image.height * ratio)
            
            resized_img = el.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            x_pos = (target_w - new_w) // 2
            
            processed_elements.append((resized_img, x_pos, current_y))
            current_y += new_h + padding_y
            
    else:
        # RECTANGLE / SQUARE
        logger.info("Layout Strategy: Stacked Center")
        total_content_height = 0
        temp_imgs = []
        
        for el in foreground:
            max_w = int(target_w * 0.85)
            ratio = min(1.0, max_w / el.image.width) if el.image.width > 0 else 1
            new_w = int(el.image.width * ratio)
            new_h = int(el.image.height * ratio)
            
            resized_img = el.image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            temp_imgs.append(resized_img)
            total_content_height += new_h + padding_y
            
        current_y = (target_h - total_content_height) // 2
        for img in temp_imgs:
            x_pos = (target_w - img.width) // 2
            processed_elements.append((img, x_pos, current_y))
            current_y += img.height + padding_y

    return processed_elements

def render_asset(background_img: Image.Image, foreground_data: list, target_size: tuple, output_path: str):
    """
    Composites the final image.
    """
    canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))
    
    if background_img:
        if background_img.mode != 'RGBA':
            background_img = background_img.convert('RGBA')
        canvas.paste(background_img, (0, 0))

    for img, x, y in foreground_data:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        canvas.paste(img, (x, y), img)
            
    canvas.convert("RGB").save(output_path, quality=95)
    logger.info(f"Saved asset to {output_path}")

def parse_psd_layers(psd_path):
    return PSDImage.open(psd_path)