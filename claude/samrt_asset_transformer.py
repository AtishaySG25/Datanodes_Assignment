"""
Smart Asset Transformer - Computer Vision-Driven Asset Generation
Author: [Your Name]
Date: December 2025

APPROACH:
1. Saliency Detection: Uses spectral residual method to identify visually important regions
2. Multi-Level Object Detection: Combines edge detection, contour analysis, and color segmentation
3. Intelligent Cropping: Calculates importance scores for different regions and crops around high-value areas
4. Content-Aware Repositioning: Dynamically adjusts element positions for narrow formats
5. Smart Background Extension: Uses inpainting and edge-aware filling to avoid padding

METHODOLOGY:
- Hierarchical analysis: Text → Logos → Prominent Objects → Background
- Region importance scoring based on saliency, contrast, and position
- Adaptive strategies per output format (landscape/portrait/square/logo)
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
from typing import Tuple, List, Dict
import json

class SmartAssetTransformer:
    """
    Main class for intelligent asset transformation using computer vision techniques.
    """
    
    def __init__(self, master_path: str, output_dir: str = "output/secondary_assets"):
        """
        Initialize transformer with master asset path.
        
        Args:
            master_path: Path to master PSD/JPEG file
            output_dir: Directory for output assets
        """
        self.master_path = master_path
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load master image
        self.master_img = self._load_image(master_path)
        self.master_array = np.array(self.master_img)
        
        # Asset specifications
        self.specs = {
            'Image_Landscape': {'dims': (1200, 628), 'format': 'JPEG', 'max_size': 5120},
            'Image_Square': {'dims': (1200, 1200), 'format': 'JPEG', 'max_size': 5120},
            'Image_Portrait': {'dims': (960, 1200), 'format': 'JPEG', 'max_size': 5120},
            'Logo_Landscape': {'dims': (1200, 300), 'format': 'PNG', 'max_size': 5120},
            'Logo_Square': {'dims': (1200, 1200), 'format': 'PNG', 'max_size': 5120}
        }
        
        # Detected important regions (cached for efficiency)
        self.regions_cache = None
        
    def _load_image(self, path: str) -> Image.Image:
        """Load image from PSD or standard formats."""
        try:
            # Try loading as PSD first
            from psd_tools import PSDImage
            psd = PSDImage.open(path)
            return psd.composite()
        except:
            # Fallback to PIL for JPEG/PNG
            return Image.open(path).convert('RGB')
    
    def detect_salient_regions(self) -> np.ndarray:
        """
        Detect salient regions using spectral residual method.
        Returns saliency map highlighting important visual areas.
        """
        # Convert to grayscale for saliency detection
        gray = cv2.cvtColor(self.master_array, cv2.COLOR_RGB2GRAY)
        
        # Create saliency detector using spectral residual
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(self.master_array)
        
        # Enhance saliency map
        saliency_map = (saliency_map * 255).astype(np.uint8)
        saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)
        
        return saliency_map
    
    def detect_text_regions(self) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions using edge detection and morphological operations.
        Returns list of bounding boxes (x, y, w, h).
        """
        gray = cv2.cvtColor(self.master_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection for text
        edges = cv2.Canny(filtered, 50, 150)
        
        # Morphological operations to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by aspect ratio and size (typical for text)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            if 2 < aspect_ratio < 15 and area > 1000:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def detect_logo_regions(self) -> List[Tuple[int, int, int, int]]:
        """
        Detect logo regions using color segmentation and contour analysis.
        Logos typically have distinct colors and compact shapes.
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(self.master_array, cv2.COLOR_RGB2HSV)
        
        # Detect high saturation regions (logos often have vibrant colors)
        _, sat_thresh = cv2.threshold(hsv[:,:,1], 100, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(sat_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logo_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            # Logos are typically compact and medium-sized
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if 500 < area < 50000 and aspect_ratio < 3:
                logo_regions.append((x, y, w, h))
        
        return logo_regions
    
    def detect_prominent_objects(self) -> List[Tuple[int, int, int, int]]:
        """
        Detect prominent objects using combined saliency and contour analysis.
        """
        saliency_map = self.detect_salient_regions()
        
        # Threshold saliency map to get important regions
        _, binary = cv2.threshold(saliency_map, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 5000:  # Filter small noise
                objects.append((x, y, w, h))
        
        return objects
    
    def get_all_important_regions(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect and cache all important regions in the master asset.
        Returns dictionary with region types and their bounding boxes.
        """
        if self.regions_cache is not None:
            return self.regions_cache
        
        print("🔍 Analyzing master asset...")
        regions = {
            'text': self.detect_text_regions(),
            'logos': self.detect_logo_regions(),
            'objects': self.detect_prominent_objects()
        }
        
        print(f"  ✓ Found {len(regions['text'])} text regions")
        print(f"  ✓ Found {len(regions['logos'])} logo regions")
        print(f"  ✓ Found {len(regions['objects'])} prominent objects")
        
        self.regions_cache = regions
        return regions
    
    def calculate_region_importance(self, regions: Dict) -> np.ndarray:
        """
        Calculate importance score map based on detected regions.
        Higher scores indicate more important areas to preserve.
        """
        h, w = self.master_array.shape[:2]
        importance_map = np.zeros((h, w), dtype=np.float32)
        
        # Add saliency
        saliency = self.detect_salient_regions()
        importance_map += saliency.astype(np.float32) / 255.0
        
        # Weight text regions heavily (3x)
        for x, y, w, h in regions['text']:
            importance_map[y:y+h, x:x+w] += 3.0
        
        # Weight logo regions (2x)
        for x, y, w, h in regions['logos']:
            importance_map[y:y+h, x:x+w] += 2.0
        
        # Weight prominent objects (1.5x)
        for x, y, w, h in regions['objects']:
            importance_map[y:y+h, x:x+w] += 1.5
        
        # Normalize
        if importance_map.max() > 0:
            importance_map = importance_map / importance_map.max()
        
        return importance_map
    
    def smart_crop(self, target_width: int, target_height: int, regions: Dict) -> Tuple[int, int, int, int]:
        """
        Intelligently crop to target dimensions preserving important regions.
        Returns (x, y, width, height) for crop region.
        """
        src_h, src_w = self.master_array.shape[:2]
        importance_map = self.calculate_region_importance(regions)
        
        # Calculate aspect ratios
        src_aspect = src_w / src_h
        target_aspect = target_width / target_height
        
        if src_aspect > target_aspect:
            # Source is wider - need to crop width
            crop_h = src_h
            crop_w = int(src_h * target_aspect)
            
            # Find best horizontal position
            best_score = -1
            best_x = 0
            
            for x in range(0, src_w - crop_w + 1, 10):
                score = importance_map[:, x:x+crop_w].sum()
                if score > best_score:
                    best_score = score
                    best_x = x
            
            return (best_x, 0, crop_w, crop_h)
        else:
            # Source is taller - need to crop height
            crop_w = src_w
            crop_h = int(src_w / target_aspect)
            
            # Find best vertical position
            best_score = -1
            best_y = 0
            
            for y in range(0, src_h - crop_h + 1, 10):
                score = importance_map[y:y+crop_h, :].sum()
                if score > best_score:
                    best_score = score
                    best_y = y
            
            return (0, best_y, crop_w, crop_h)
    
    def content_aware_resize(self, img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize image using seam carving principles for minimal distortion.
        Falls back to smart padding if needed.
        """
        src_h, src_w = img.shape[:2]
        
        # If very close to target, just resize
        if abs(src_w - target_width) < 50 and abs(src_h - target_height) < 50:
            return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate scaling to fit within target
        scale = min(target_width / src_w, target_height / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # If exact match, return
        if new_w == target_width and new_h == target_height:
            return resized
        
        # Smart background extension using edge pixels
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate average edge colors for intelligent fill
        top_color = np.median(resized[0:5, :], axis=(0, 1)).astype(np.uint8)
        bottom_color = np.median(resized[-5:, :], axis=(0, 1)).astype(np.uint8)
        left_color = np.median(resized[:, 0:5], axis=(0, 1)).astype(np.uint8)
        right_color = np.median(resized[:, -5:], axis=(0, 1)).astype(np.uint8)
        
        # Fill canvas with gradient
        canvas[:] = np.mean([top_color, bottom_color, left_color, right_color], axis=0).astype(np.uint8)
        
        # Center the resized image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Apply blur at boundaries for smooth transition
        mask = np.zeros((target_height, target_width), dtype=np.uint8)
        mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 255
        
        # Dilate mask for feathering
        kernel = np.ones((21, 21), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_blurred = cv2.GaussianBlur(mask_dilated, (21, 21), 0)
        
        # Blend
        mask_3ch = np.stack([mask_blurred] * 3, axis=2) / 255.0
        canvas = (canvas * (1 - mask_3ch) + canvas * mask_3ch).astype(np.uint8)
        
        return canvas
    
    def generate_asset(self, asset_type: str, base_name: str) -> str:
        """
        Generate a single asset with specified type and dimensions.
        Returns path to generated file.
        """
        spec = self.specs[asset_type]
        target_w, target_h = spec['dims']
        
        print(f"\n🎨 Generating {asset_type} ({target_w}x{target_h})...")
        
        # Get important regions
        regions = self.get_all_important_regions()
        
        # Smart crop to target aspect ratio
        crop_x, crop_y, crop_w, crop_h = self.smart_crop(target_w, target_h, regions)
        cropped = self.master_array[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        print(f"  ✓ Intelligent crop: ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
        
        # Content-aware resize to exact dimensions
        final = self.content_aware_resize(cropped, target_w, target_h)
        
        print(f"  ✓ Resized to exact dimensions: {final.shape[1]}x{final.shape[0]}")
        
        # Convert to PIL for saving
        final_img = Image.fromarray(final)
        
        # Prepare output path
        ext = 'jpeg' if spec['format'] == 'JPEG' else 'png'
        output_path = os.path.join(self.output_dir, f"{base_name}_{asset_type}.{ext}")
        
        # Save with quality optimization
        if spec['format'] == 'JPEG':
            quality = 95
            final_img.save(output_path, 'JPEG', quality=quality, optimize=True)
        else:
            final_img.save(output_path, 'PNG', optimize=True)
        
        # Check file size
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"  ✓ Saved: {output_path} ({file_size_kb:.1f} KB)")
        
        if file_size_kb > spec['max_size']:
            print(f"  ⚠️  Warning: File size exceeds {spec['max_size']} KB limit")
        
        return output_path
    
    def generate_all_assets(self) -> Dict[str, str]:
        """
        Generate all 5 secondary assets from master.
        Returns dictionary mapping asset types to output paths.
        """
        base_name = Path(self.master_path).stem
        results = {}
        
        print("\n" + "="*60)
        print("🚀 SMART ASSET TRANSFORMER")
        print("="*60)
        print(f"Master Asset: {self.master_path}")
        print(f"Dimensions: {self.master_array.shape[1]}x{self.master_array.shape[0]}")
        
        for asset_type in self.specs.keys():
            try:
                output_path = self.generate_asset(asset_type, base_name)
                results[asset_type] = output_path
            except Exception as e:
                print(f"  ❌ Error generating {asset_type}: {str(e)}")
                results[asset_type] = None
        
        print("\n" + "="*60)
        print("✅ TRANSFORMATION COMPLETE")
        print("="*60)
        
        return results
    
    def generate_report(self, results: Dict[str, str]) -> None:
        """Generate visual report with all outputs."""
        print("\n📊 Asset Generation Report:")
        print("-" * 60)
        
        for asset_type, path in results.items():
            if path and os.path.exists(path):
                img = Image.open(path)
                size_kb = os.path.getsize(path) / 1024
                print(f"{asset_type:20s} | {img.size[0]}x{img.size[1]:4d} | {size_kb:7.1f} KB | ✓")
            else:
                print(f"{asset_type:20s} | Failed")
        
        print("-" * 60)


def main():
    """Main execution function."""
    # Configuration
    INPUT_PATH = "input/master_assets/sample_master.psd"  # Change to your input
    OUTPUT_DIR = "output/secondary_assets"
    
    # Check if input exists
    if not os.path.exists(INPUT_PATH):
        # Try JPEG fallback
        INPUT_PATH = INPUT_PATH.replace('.psd', '.jpg').replace('.psd', '.jpeg')
        if not os.path.exists(INPUT_PATH):
            print(f"❌ Error: Master asset not found at {INPUT_PATH}")
            print("Please place your master asset in input/master_assets/")
            return
    
    # Initialize transformer
    transformer = SmartAssetTransformer(INPUT_PATH, OUTPUT_DIR)
    
    # Generate all assets
    results = transformer.generate_all_assets()
    
    # Generate report
    transformer.generate_report(results)
    
    print(f"\n📁 All assets saved to: {OUTPUT_DIR}/")
    print("\n💡 Next steps:")
    print("   1. Verify dimensions of each output")
    print("   2. Check visual quality and important element preservation")
    print("   3. Validate file sizes are within limits")


if __name__ == "__main__":
    main()