# Asset Transformation Tool
# Author: [Your Name]
# Date: December 22, 2025
# Description: A scalable Python tool for transforming a master PSD/JPEG asset into secondary formats using
#              computer vision techniques. Demonstrates PSD parsing, saliency-based object detection,
#              smart cropping via saliency maximization, and object repositioning for aspect ratio adaptation.
#
# Installation:
# pip install psd-tools opencv-python pillow numpy scikit-learn
#
# Usage:
# python asset_transformer.py --input input/master_assets/sample_master.psd --output output/secondary_assets/
#
# Core CV Approach (3-sentence summary):
# 1. Object Identification: Uses custom high-contrast edge + color saliency to reliably detect bold text, logos, and subjects.
# 2. Smart Cropping: Maximizes salient content in the crop window using integral images for efficient search.
# 3. Object Repositioning: Segments key elements with refined GrabCut, then intelligently recomposes them on the dominant background
#    color for logo formats – horizontal layout for narrow banners, bold centered placement for square logos.
#
# Scalability Notes: Modular design supports batch processing; future extensions can incorporate deep learning detectors.

import argparse
import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
import os
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetTransformer:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.master_img = None
        self.integral_saliency = None
        self.key_objects = []
        self.dominant_color = None

    def load_master_asset(self):
        """Load PSD or JPEG as composite RGB image."""
        if self.input_path.lower().endswith('.psd'):
            psd = PSDImage.open(self.input_path)
            pil_img = psd.composite()
            logger.info(f"Loaded PSD composite: {pil_img.size}")
        else:
            pil_img = Image.open(self.input_path)
            logger.info(f"Loaded JPEG: {pil_img.size}")

        self.master_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if self.master_img.shape[:2] != (1080, 1080):
            logger.warning("Input is not 1080x1080; resizing to match spec.")
            self.master_img = cv2.resize(self.master_img, (1080, 1080), interpolation=cv2.INTER_LANCZOS4)
        return self.master_img

    def detect_key_objects(self):
        """Robust custom saliency emphasizing text, faces, and high-contrast objects."""
        gray = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for text/logos
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=4)
        edges = cv2.GaussianBlur(edges, (31, 31), 0)
        
        # Color saliency in Lab space
        lab = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        color_sal = cv2.GaussianBlur(a**2 + b**2, (31, 31), 0)
        
        # Combine – strong weight on edges for bold text
        saliency_map = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_map = cv2.normalize(color_sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        saliency_map = cv2.addWeighted(saliency_map, 0.8, color_map, 0.2, 0)
        
        # Otsu threshold + morphological cleanup
        _, binary_mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
        
        self.integral_saliency = cv2.integral(binary_mask)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = self.master_img.shape[:2]
        img_area = img_h * img_w
        
        # Scoring function – favor large + upper elements
        def score(cnt):
            area = cv2.contourArea(cnt)
            _, y, _, h = cv2.boundingRect(cnt)
            return area * (1 + 1.0 * (1 - y / img_h))
        
        candidates = sorted(contours, key=score, reverse=True)[:6]
        
        self.key_objects = []
        for cnt in candidates:
            area = cv2.contourArea(cnt)
            if area < 0.01 * img_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            mask = self._improved_grabcut(x, y, w, h, margin=30)
            self.key_objects.append(((x, y, w, h), mask))
        
        # Dominant background color from non-salient regions
        mask_inv = cv2.bitwise_not(binary_mask)
        bg_pixels = cv2.bitwise_and(self.master_img, self.master_img, mask=mask_inv)
        valid_pixels = bg_pixels.reshape(-1, 3)
        valid_pixels = valid_pixels[np.any(valid_pixels != [0, 0, 0], axis=1)]
        
        if len(valid_pixels) > 100:
            kmeans = KMeans(n_clusters=1, n_init=10, random_state=42).fit(valid_pixels)
            self.dominant_color = tuple(map(int, kmeans.cluster_centers_[0]))
        else:
            # Fallback: average from corners
            corners = np.vstack([
                self.master_img[:50, :50],
                self.master_img[:50, -50:],
                self.master_img[-50:, :50],
                self.master_img[-50:, -50:]
            ]).reshape(-1, 3)
            self.dominant_color = tuple(map(int, np.mean(corners, axis=0)))
        
        logger.info(f"Dominant background color: {self.dominant_color} | Detected {len(self.key_objects)} key objects")

    def _improved_grabcut(self, x, y, w, h, margin=30):
        """Accurate GrabCut segmentation with expanded rectangle and cleanup."""
        img_h, img_w = self.master_img.shape[:2]
        rect_x = max(0, x - margin)
        rect_y = max(0, y - margin)
        rect_w = min(img_w - rect_x, w + 2 * margin)
        rect_h = min(img_h - rect_y, h + 2 * margin)
        rect = (rect_x, rect_y, rect_w, rect_h)
        
        mask = np.zeros(self.master_img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(self.master_img, mask, rect, bgd_model, fgd_model, iterCount=15, mode=cv2.GC_INIT_WITH_RECT)
        
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
        
        return mask

    def _paste_with_mask(self, canvas, obj, mask, x, y):
        """Clean alpha blending for object pasting."""
        roi = canvas[y:y+obj.shape[0], x:x+obj.shape[1]]
        fg = cv2.bitwise_and(obj, obj, mask=mask)
        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
        canvas[y:y+obj.shape[0], x:x+obj.shape[1]] = cv2.add(fg, bg)

    # Fixed the unpacking error in reposition_objects (Logo_Landscape branch)

# Replace the entire reposition_objects method with this corrected version:

    def reposition_objects(self, target_shape, is_logo=False):
        """Professional logo composition on dominant background."""
        h, w = target_shape[:2]
        canvas = np.full((h, w, 3), self.dominant_color, dtype=np.uint8)
        
        if not self.key_objects:
            logger.warning("No key objects detected – falling back to smart crop")
            x, y, crop_w, crop_h = self.smart_crop_coordinates(w, h)
            cropped = self.master_img[y:y+crop_h, x:x+crop_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Sort by importance
        sorted_objs = sorted(self.key_objects,
                             key=lambda o: (o[0][2] * o[0][3]) * (1 + 1.0 * (1 - o[0][1] / self.master_img.shape[0])),
                             reverse=True)
        
        if is_logo and w >= 3 * h:  # Logo_Landscape (1200x300) – horizontal layout
            max_height = int(h * 0.80)
            objs_to_use = sorted_objs[:3]
            
            scales = [max_height / oh for _, _, _, oh in [obj[0] for obj in objs_to_use]]
            new_sizes = [(int(ow * s), int(oh * s)) for (ow, oh), s in zip([obj[0][2:4] for obj in objs_to_use], scales)]
            total_content_w = sum(nw for nw, _ in new_sizes)
            spacing = max(40, (w - total_content_w) // (len(new_sizes) + 1))
            
            current_x = spacing
            # === FIXED LOOP: use enumerate to get index i separately ===
            for i, ((ox, oy, ow, oh), mask) in enumerate(objs_to_use):
                new_w, new_h = new_sizes[i]
                obj_crop = self.master_img[oy:oy+oh, ox:ox+ow]
                mask_crop = mask[oy:oy+oh, ox:ox+ow]
                
                resized_obj = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                resized_mask = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                resized_mask = (resized_mask > 128).astype(np.uint8)
                
                y_pos = (h - new_h) // 2
                self._paste_with_mask(canvas, resized_obj, resized_mask, current_x, y_pos)
                current_x += new_w + spacing
                
        else:  # Logo_Square and others – bold centered composition
            (ox, oy, ow, oh), mask = sorted_objs[0]
            max_dim = min(w * 0.92, h * 0.92)
            scale = max_dim / max(ow, oh)
            new_w, new_h = int(ow * scale), int(oh * scale)
            
            obj_crop = self.master_img[oy:oy+oh, ox:ox+ow]
            mask_crop = mask[oy:oy+oh, ox:ox+ow]
            
            resized_obj = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            resized_mask = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            resized_mask = (resized_mask > 128).astype(np.uint8)
            
            x_pos = (w - new_w) // 2
            y_pos = (h - new_h) // 2
            self._paste_with_mask(canvas, resized_obj, resized_mask, x_pos, y_pos)
            
            # Optional secondary element in corner (only for larger canvases)
            if len(sorted_objs) > 1 and h > 800:
                (ox2, oy2, ow2, oh2), mask2 = sorted_objs[1]
                small_scale = 0.4
                sw, sh = int(ow2 * small_scale), int(oh2 * small_scale)
                if sw < w * 0.3:
                    small_obj = cv2.resize(self.master_img[oy2:oy2+oh2, ox2:ox2+ow2], (sw, sh), interpolation=cv2.INTER_LANCZOS4)
                    small_mask = cv2.resize(mask2[oy2:oy2+oh2, ox2:ox2+ow2], (sw, sh), interpolation=cv2.INTER_NEAREST)
                    small_mask = (small_mask > 128).astype(np.uint8)
                    self._paste_with_mask(canvas, small_obj, small_mask, w - sw - 50, h - sh - 50)
        
        return canvas

    def smart_crop_coordinates(self, target_w, target_h, img_w=1080, img_h=1080):
        """Find optimal crop maximizing saliency."""
        aspect_target = target_w / target_h
        aspect_src = img_w / img_h

        if aspect_target > aspect_src:
            crop_h = img_h
            crop_w = int(crop_h * aspect_target)
        else:
            crop_w = img_w
            crop_h = int(crop_w / aspect_target)

        crop_w = min(crop_w, img_w)
        crop_h = min(crop_h, img_h)

        max_saliency = -1
        best_x, best_y = (img_w - crop_w) // 2, (img_h - crop_h) // 2

        step = max(10, crop_w // 50)
        for y in range(0, img_h - crop_h + 1, step):
            for x in range(0, img_w - crop_w + 1, step):
                s = (self.integral_saliency[x + crop_w, y + crop_h] -
                     self.integral_saliency[x, y + crop_h] -
                     self.integral_saliency[x + crop_w, y] +
                     self.integral_saliency[x, y])
                if s > max_saliency:
                    max_saliency = s
                    best_x, best_y = x, y

        return best_x, best_y, crop_w, crop_h

    def transform_to_format(self, target_name, target_w, target_h, format_ext, is_logo=False):
        """Generate one transformed asset."""
        if is_logo:
            transformed = self.reposition_objects((target_h, target_w), is_logo=True)
        else:
            x, y, crop_w, crop_h = self.smart_crop_coordinates(target_w, target_h)
            cropped = self.master_img[y:y + crop_h, x:x + crop_w]
            transformed = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        output_path = os.path.join(self.output_dir, f"sample_master_{target_name}.{format_ext.lower()}")
        if format_ext.lower() == 'png':
            cv2.imwrite(output_path, transformed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(output_path, transformed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        loaded = cv2.imread(output_path)
        if loaded.shape[:2] == (target_h, target_w):
            logger.info(f"Generated {target_name}: {target_w}x{target_h} ({os.path.getsize(output_path)/1024:.1f} KB)")
        else:
            logger.error(f"Dimension mismatch for {target_name}")

        return output_path

    def run(self):
        """Full pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_master_asset()
        self.detect_key_objects()

        formats = [
            ("Image_Landscape", 1200, 628, "jpeg", False),
            ("Image_Square", 1200, 1200, "jpeg", False),
            ("Image_Portrait", 960, 1200, "jpeg", False),
            ("Logo_Landscape", 1200, 300, "png", True),
            ("Logo_Square", 1200, 1200, "png", True),
        ]

        outputs = []
        for name, w, h, ext, is_logo in formats:
            path = self.transform_to_format(name, w, h, ext, is_logo)
            outputs.append(path)

        logger.info("All transformations complete.")
        return outputs

def main():
    parser = argparse.ArgumentParser(description="Asset Transformation Tool")
    parser.add_argument("--input", required=True, help="Path to master PSD/JPEG")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    transformer = AssetTransformer(args.input, args.output)
    outputs = transformer.run()

    print("\nGenerated assets:")
    for path in outputs:
        print(f"- {path}")

if __name__ == "__main__":
    main()