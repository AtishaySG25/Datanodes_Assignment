"""
Logo Square Generator - Simplified Script
Generates only the Logo_Square asset (1200x1200 PNG) from master asset

Usage:
    python generate_logo_square.py
    python generate_logo_square.py --input path/to/master.psd --output path/to/output.png
"""

import cv2
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

class LogoSquareGenerator:
    """Generate Logo_Square asset using computer vision techniques."""
    
    def __init__(self, master_path: str):
        """
        Initialize generator with master asset path.
        
        Args:
            master_path: Path to master PSD/JPEG file
        """
        self.master_path = master_path
        self.master_img = self._load_image(master_path)
        self.master_array = np.array(self.master_img)
        
        # Logo Square specifications
        self.target_width = 1200
        self.target_height = 1200
        
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
        gray = cv2.cvtColor(self.master_array, cv2.COLOR_RGB2GRAY)
        
        # Create saliency detector
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
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection
        edges = cv2.Canny(filtered, 50, 150)
        
        # Morphological operations to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            if 2 < aspect_ratio < 15 and area > 1000:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def detect_logo_regions(self) -> List[Tuple[int, int, int, int]]:
        """
        Detect logo regions using color segmentation and contour analysis.
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(self.master_array, cv2.COLOR_RGB2HSV)
        
        # Detect high saturation regions
        _, sat_thresh = cv2.threshold(hsv[:,:,1], 100, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(sat_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logo_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if 500 < area < 50000 and aspect_ratio < 3:
                logo_regions.append((x, y, w, h))
        
        return logo_regions
    
    def detect_prominent_objects(self) -> List[Tuple[int, int, int, int]]:
        """
        Detect prominent objects using combined saliency and contour analysis.
        """
        saliency_map = self.detect_salient_regions()
        
        # Threshold saliency map
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
            if area > 5000:
                objects.append((x, y, w, h))
        
        return objects
    
    def get_all_important_regions(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Detect all important regions in the master asset.
        """
        print("🔍 Analyzing master asset for important regions...")
        regions = {
            'text': self.detect_text_regions(),
            'logos': self.detect_logo_regions(),
            'objects': self.detect_prominent_objects()
        }
        
        print(f"  ✓ Found {len(regions['text'])} text regions")
        print(f"  ✓ Found {len(regions['logos'])} logo regions")
        print(f"  ✓ Found {len(regions['objects'])} prominent objects")
        
        return regions
    
    def calculate_region_importance(self, regions: Dict) -> np.ndarray:
        """
        Calculate importance score map based on detected regions.
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
    
    def smart_crop_square(self, regions: Dict) -> Tuple[int, int, int, int]:
        """
        Intelligently crop to square (1200x1200) preserving important regions.
        For square output from square input, this finds the best centered region.
        Returns (x, y, width, height) for crop region.
        """
        src_h, src_w = self.master_array.shape[:2]
        importance_map = self.calculate_region_importance(regions)
        
        # For square to square, we want to preserve the center while maximizing importance
        # Since both are square (1080x1080 -> 1200x1200), we'll use the full image
        # and resize up, but let's find the optimal square crop within the source
        
        crop_size = min(src_h, src_w)
        
        # Find best square crop position
        best_score = -1
        best_x = 0
        best_y = 0
        
        # Check different positions (step by 10 for efficiency)
        for y in range(0, src_h - crop_size + 1, 10):
            for x in range(0, src_w - crop_size + 1, 10):
                score = importance_map[y:y+crop_size, x:x+crop_size].sum()
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_y = y
        
        return (best_x, best_y, crop_size, crop_size)
    
    def content_aware_resize(self, img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize image intelligently to exact target dimensions.
        """
        src_h, src_w = img.shape[:2]
        
        # Calculate scaling to fit within target
        scale = min(target_width / src_w, target_height / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # If exact match, return
        if new_w == target_width and new_h == target_height:
            return resized
        
        # Smart background extension
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate average edge colors
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
        
        return canvas
    
    def generate(self, output_path: str = None) -> str:
        """
        Generate Logo_Square (1200x1200 PNG) asset.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to generated file
        """
        if output_path is None:
            base_name = Path(self.master_path).stem
            output_path = f"claude_Logo_Square.png"
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("🎨 LOGO SQUARE GENERATOR")
        print("="*60)
        print(f"Master Asset: {self.master_path}")
        print(f"Dimensions: {self.master_array.shape[1]}x{self.master_array.shape[0]}")
        print(f"Target: {self.target_width}x{self.target_height} PNG")
        
        # Get important regions
        regions = self.get_all_important_regions()
        
        # Smart crop to square
        print(f"\n🔍 Finding optimal square crop...")
        crop_x, crop_y, crop_w, crop_h = self.smart_crop_square(regions)
        cropped = self.master_array[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        print(f"  ✓ Crop region: ({crop_x}, {crop_y}, {crop_w}, {crop_h})")
        
        # Resize to target dimensions
        print(f"\n🎨 Resizing to {self.target_width}x{self.target_height}...")
        final = self.content_aware_resize(cropped, self.target_width, self.target_height)
        print(f"  ✓ Final dimensions: {final.shape[1]}x{final.shape[0]}")
        
        # Convert to PIL and save as PNG
        final_img = Image.fromarray(final)
        final_img.save(output_path, 'PNG', optimize=True)
        
        # Check file size
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"\n✅ Logo Square generated successfully!")
        print(f"  📁 Output: {output_path}")
        print(f"  📏 Dimensions: {final.shape[1]}x{final.shape[0]}")
        print(f"  💾 File size: {file_size_kb:.1f} KB")
        
        if file_size_kb > 5120:
            print(f"  ⚠️  Warning: File size exceeds 5120 KB limit")
        
        print("="*60 + "\n")
        
        return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate Logo_Square (1200x1200 PNG) from master asset'
    )
    parser.add_argument(
        '--input', '-i',
        default='D:/Datanodes_Assignment/lady.psd',
        help='Path to master asset (PSD/JPEG/PNG)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output path (default: output/secondary_assets/claude_Logo_Square.png)'
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        # Try alternative extensions
        base_path = args.input.rsplit('.', 1)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.psd']:
            alt_path = base_path + ext
            if os.path.exists(alt_path):
                args.input = alt_path
                break
        else:
            print(f"❌ Error: Master asset not found at {args.input}")
            print("Please provide a valid path using --input")
            return 1
    
    try:
        # Initialize generator
        generator = LogoSquareGenerator(args.input)
        
        # Generate logo square
        output_path = generator.generate(args.output)
        
        print("💡 Tip: Use test_outputs.py to validate dimensions and format")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error generating Logo Square: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())