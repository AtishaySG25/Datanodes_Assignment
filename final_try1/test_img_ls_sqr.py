# # Minimal Asset Transformation Script
# # Focus: Generate only Image_Landscape (1200x628 JPEG) and Image_Square (1200x1200 JPEG)
# # Using smart saliency-based cropping from a 1080x1080 master asset (PSD or JPEG)

# # Installation required:
# # pip install psd-tools opencv-python pillow numpy scikit-learn

# import cv2
# import numpy as np
# from PIL import Image
# from psd_tools import PSDImage
# import os
# import logging
# from sklearn.cluster import KMeans

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ImageCropper:
#     def __init__(self, input_path, output_dir):
#         self.input_path = input_path
#         self.output_dir = output_dir
#         self.master_img = None  # BGR format
#         self.integral_saliency = None

#     def load_master_asset(self):
#         """Load PSD or JPEG and resize to 1080x1080 if needed."""
#         if self.input_path.lower().endswith('.psd'):
#             psd = PSDImage.open(self.input_path)
#             pil_img = psd.composite()
#             logger.info(f"Loaded PSD: {pil_img.size}")
#         else:
#             pil_img = Image.open(self.input_path)
#             logger.info(f"Loaded image: {pil_img.size}")

#         self.master_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#         if self.master_img.shape[:2] != (1080, 1080):
#             logger.warning("Resizing master to 1080x1080")
#             self.master_img = cv2.resize(self.master_img, (1080, 1080), interpolation=cv2.INTER_LANCZOS4)

#     def compute_saliency_mask(self):
#         """Custom saliency emphasizing text, high-contrast objects, and color distinctiveness."""
#         gray = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2GRAY)
        
#         # Strong edge detection for text and logos
#         edges = cv2.Canny(gray, 80, 160)
#         edges = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=4)
#         edges = cv2.GaussianBlur(edges, (31, 31), 0)
        
#         # Color saliency in Lab space
#         lab = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2Lab)
#         l, a, b = cv2.split(lab)
#         color_sal = cv2.GaussianBlur(a**2 + b**2, (31, 31), 0)
        
#         # Combine with emphasis on edges
#         saliency_map = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         color_map = cv2.normalize(color_sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#         saliency_map = cv2.addWeighted(saliency_map, 0.8, color_map, 0.2, 0)
        
#         # Threshold and clean up
#         _, binary_mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
#         binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
        
#         self.integral_saliency = cv2.integral(binary_mask)
#         logger.info("Saliency mask computed")

#     def smart_crop_coordinates(self, target_w, target_h, img_w=1080, img_h=1080):
#         """Find the crop window that maximizes salient content using integral image."""
#         aspect_target = target_w / target_h
#         aspect_src = img_w / img_h

#         if aspect_target > aspect_src:  # Landscape: crop height
#             crop_h = img_h
#             crop_w = int(crop_h * aspect_target)
#         else:  # Square or portrait: crop width
#             crop_w = img_w
#             crop_h = int(crop_w / aspect_target)

#         crop_w = min(crop_w, img_w)
#         crop_h = min(crop_h, img_h)

#         max_saliency = -1
#         best_x, best_y = (img_w - crop_w) // 2, (img_h - crop_h) // 2

#         step = max(10, crop_w // 50)
#         for y in range(0, img_h - crop_h + 1, step):
#             for x in range(0, img_w - crop_w + 1, step):
#                 s = (self.integral_saliency[x + crop_w, y + crop_h] -
#                      self.integral_saliency[x, y + crop_h] -
#                      self.integral_saliency[x + crop_w, y] +
#                      self.integral_saliency[x, y])
#                 if s > max_saliency:
#                     max_saliency = s
#                     best_x, best_y = x, y

#         logger.info(f"Best crop for {target_w}x{target_h}: ({best_x}, {best_y}, {crop_w}x{crop_h})")
#         return best_x, best_y, crop_w, crop_h

#     def create_landscape(self):
#         """Generate Image_Landscape: 1200x628 JPEG"""
#         x, y, crop_w, crop_h = self.smart_crop_coordinates(1200, 628)
#         cropped = self.master_img[y:y + crop_h, x:x + crop_w]
#         result = cv2.resize(cropped, (1200, 628), interpolation=cv2.INTER_LANCZOS4)
        
#         os.makedirs(self.output_dir, exist_ok=True)
#         output_path = os.path.join(self.output_dir, "sample_master_Image_Landscape.jpeg")
#         cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         logger.info(f"Image_Landscape saved: {output_path}")
#         return output_path

#     def create_square(self):
#         """Generate Image_Square: 1200x1200 JPEG"""
#         x, y, crop_w, crop_h = self.smart_crop_coordinates(1200, 1200)
#         cropped = self.master_img[y:y + crop_h, x:x + crop_w]
#         result = cv2.resize(cropped, (1200, 1200), interpolation=cv2.INTER_LANCZOS4)
        
#         os.makedirs(self.output_dir, exist_ok=True)
#         output_path = os.path.join(self.output_dir, "sample_master_Image_Square.jpeg")
#         cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         logger.info(f"Image_Square saved: {output_path}")
#         return output_path

#     def run(self):
#         """Full pipeline for Landscape and Square only."""
#         self.load_master_asset()
#         self.compute_saliency_mask()
#         self.create_landscape()
#         self.create_square()
#         logger.info("Landscape and Square assets generated successfully.")

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Generate Image_Landscape and Image_Square from master asset")
#     parser.add_argument("--input", required=True, help="Path to master PSD or JPEG (ideally 1080x1080)")
#     parser.add_argument("--output", required=True, help="Output directory for generated assets")
#     args = parser.parse_args()

#     cropper = ImageCropper(args.input, args.output)
#     cropper.run()

#     print("\nGenerated files:")
#     print("- sample_master_Image_Landscape.jpeg (1200x628)")
#     print("- sample_master_Image_Square.jpeg    (1200x1200)")

# Minimal Asset Transformation Script
# Generates ONLY the three image formats:
# - Image_Landscape (1200x628 JPEG)
# - Image_Square    (1200x1200 JPEG)
# - Image_Portrait  (960x1200 JPEG)
#
# Uses intelligent saliency-based smart cropping for all three

# Installation required:
# pip install psd-tools opencv-python pillow numpy scikit-learn

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage
import os
import logging
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCropper:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.master_img = None  # BGR format
        self.integral_saliency = None

    def load_master_asset(self):
        """Load PSD or JPEG and resize to 1080x1080 if needed."""
        if self.input_path.lower().endswith('.psd'):
            psd = PSDImage.open(self.input_path)
            pil_img = psd.composite()
            logger.info(f"Loaded PSD: {pil_img.size}")
        else:
            pil_img = Image.open(self.input_path)
            logger.info(f"Loaded image: {pil_img.size}")

        self.master_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if self.master_img.shape[:2] != (1080, 1080):
            logger.warning("Resizing master to 1080x1080")
            self.master_img = cv2.resize(self.master_img, (1080, 1080), interpolation=cv2.INTER_LANCZOS4)

    def compute_saliency_mask(self):
        """Custom saliency emphasizing text, high-contrast objects, and color distinctiveness."""
        gray = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2GRAY)
        
        # Strong edge detection for text and logos
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=4)
        edges = cv2.GaussianBlur(edges, (31, 31), 0)
        
        # Color saliency in Lab space
        lab = cv2.cvtColor(self.master_img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        color_sal = cv2.GaussianBlur(a**2 + b**2, (31, 31), 0)
        
        # Combine with emphasis on edges
        saliency_map = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_map = cv2.normalize(color_sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        saliency_map = cv2.addWeighted(saliency_map, 0.8, color_map, 0.2, 0)
        
        # Threshold and clean up
        _, binary_mask = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
        
        self.integral_saliency = cv2.integral(binary_mask)
        logger.info("Saliency mask computed")

    def smart_crop_coordinates(self, target_w, target_h, img_w=1080, img_h=1080):
        """Find the crop window that maximizes salient content using integral image."""
        aspect_target = target_w / target_h
        aspect_src = img_w / img_h

        if aspect_target > aspect_src:  # Wider than source → crop height
            crop_h = img_h
            crop_w = int(crop_h * aspect_target)
        else:  # Taller or same → crop width
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

        logger.info(f"Best crop for {target_w}x{target_h}: ({best_x}, {best_y}, {crop_w}x{crop_h})")
        return best_x, best_y, crop_w, crop_h

    def create_landscape(self):
        """Generate Image_Landscape: 1200x628 JPEG"""
        x, y, crop_w, crop_h = self.smart_crop_coordinates(1200, 628)
        cropped = self.master_img[y:y + crop_h, x:x + crop_w]
        result = cv2.resize(cropped, (1200, 628), interpolation=cv2.INTER_LANCZOS4)
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "sample_master_Image_Landscape.jpeg")
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Image_Landscape saved: {output_path}")
        return output_path

    def create_square(self):
        """Generate Image_Square: 1200x1200 JPEG"""
        x, y, crop_w, crop_h = self.smart_crop_coordinates(1200, 1200)
        cropped = self.master_img[y:y + crop_h, x:x + crop_w]
        result = cv2.resize(cropped, (1200, 1200), interpolation=cv2.INTER_LANCZOS4)
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "sample_master_Image_Square.jpeg")
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Image_Square saved: {output_path}")
        return output_path

    def create_portrait(self):
        """Generate Image_Portrait: 960x1200 JPEG"""
        x, y, crop_w, crop_h = self.smart_crop_coordinates(960, 1200)
        cropped = self.master_img[y:y + crop_h, x:x + crop_w]
        result = cv2.resize(cropped, (960, 1200), interpolation=cv2.INTER_LANCZOS4)
        
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "sample_master_Image_Portrait.jpeg")
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Image_Portrait saved: {output_path}")
        return output_path

    def run(self):
        """Full pipeline for Landscape, Square, and Portrait."""
        self.load_master_asset()
        self.compute_saliency_mask()
        self.create_landscape()
        self.create_square()
        self.create_portrait()
        logger.info("All three image assets (Landscape, Square, Portrait) generated successfully.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Image_Landscape, Image_Square, and Image_Portrait from master asset")
    parser.add_argument("--input", required=True, help="Path to master PSD or JPEG (ideally 1080x1080)")
    parser.add_argument("--output", required=True, help="Output directory for generated assets")
    args = parser.parse_args()

    cropper = ImageCropper(args.input, args.output)
    cropper.run()

    print("\nGenerated files:")
    print("- sample_master_Image_Landscape.jpeg (1200x628)")
    print("- sample_master_Image_Square.jpeg    (1200x1200)")
    print("- sample_master_Image_Portrait.jpeg  (960x1200)")
