"""
Output Validation Script
Automatically verifies all generated assets meet specifications
"""

import os
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple

class AssetValidator:
    """Validates generated assets against specifications."""
    
    def __init__(self, output_dir: str = "output/secondary_assets"):
        self.output_dir = output_dir
        self.specs = {
            'Image_Landscape': {
                'dims': (1200, 628),
                'format': 'JPEG',
                'max_size_kb': 5120,
                'extension': 'jpeg'
            },
            'Image_Square': {
                'dims': (1200, 1200),
                'format': 'JPEG',
                'max_size_kb': 5120,
                'extension': 'jpeg'
            },
            'Image_Portrait': {
                'dims': (960, 1200),
                'format': 'JPEG',
                'max_size_kb': 5120,
                'extension': 'jpeg'
            },
            'Logo_Landscape': {
                'dims': (1200, 300),
                'format': 'PNG',
                'max_size_kb': 5120,
                'extension': 'png'
            },
            'Logo_Square': {
                'dims': (1200, 1200),
                'format': 'PNG',
                'max_size_kb': 5120,
                'extension': 'png'
            }
        }
    
    def validate_dimensions(self, img: Image.Image, expected: Tuple[int, int]) -> bool:
        """Check if image dimensions match exactly."""
        return img.size == expected
    
    def validate_format(self, img: Image.Image, expected: str) -> bool:
        """Check if image format matches."""
        return img.format == expected
    
    def validate_file_size(self, path: str, max_size_kb: int) -> Tuple[bool, float]:
        """Check if file size is within limit."""
        size_kb = os.path.getsize(path) / 1024
        return size_kb <= max_size_kb, size_kb
    
    def validate_single_asset(self, asset_type: str, base_name: str) -> Dict:
        """Validate a single asset."""
        spec = self.specs[asset_type]
        ext = spec['extension']
        path = os.path.join(self.output_dir, f"{base_name}_{asset_type}.{ext}")
        
        results = {
            'exists': False,
            'dimensions_ok': False,
            'format_ok': False,
            'size_ok': False,
            'actual_dims': None,
            'actual_format': None,
            'actual_size_kb': None
        }
        
        # Check if file exists
        if not os.path.exists(path):
            return results
        
        results['exists'] = True
        
        try:
            # Open image
            img = Image.open(path)
            
            # Validate dimensions
            results['actual_dims'] = img.size
            results['dimensions_ok'] = self.validate_dimensions(img, spec['dims'])
            
            # Validate format
            results['actual_format'] = img.format
            results['format_ok'] = self.validate_format(img, spec['format'])
            
            # Validate file size
            size_ok, size_kb = self.validate_file_size(path, spec['max_size_kb'])
            results['size_ok'] = size_ok
            results['actual_size_kb'] = size_kb
            
        except Exception as e:
            print(f"Error validating {asset_type}: {str(e)}")
        
        return results
    
    def validate_all_assets(self, base_name: str = "sample_master") -> None:
        """Validate all generated assets and print report."""
        print("\n" + "="*70)
        print("📋 ASSET VALIDATION REPORT")
        print("="*70)
        
        all_passed = True
        
        for asset_type, spec in self.specs.items():
            print(f"\n🔍 {asset_type}")
            print("-" * 70)
            
            results = self.validate_single_asset(asset_type, base_name)
            
            # Exists check
            if not results['exists']:
                print(f"  ❌ File not found")
                all_passed = False
                continue
            
            print(f"  ✅ File exists")
            
            # Dimensions check
            if results['dimensions_ok']:
                print(f"  ✅ Dimensions: {results['actual_dims'][0]}×{results['actual_dims'][1]} (Expected: {spec['dims'][0]}×{spec['dims'][1]})")
            else:
                print(f"  ❌ Dimensions: {results['actual_dims'][0]}×{results['actual_dims'][1]} (Expected: {spec['dims'][0]}×{spec['dims'][1]})")
                all_passed = False
            
            # Format check
            if results['format_ok']:
                print(f"  ✅ Format: {results['actual_format']} (Expected: {spec['format']})")
            else:
                print(f"  ❌ Format: {results['actual_format']} (Expected: {spec['format']})")
                all_passed = False
            
            # Size check
            if results['size_ok']:
                print(f"  ✅ File size: {results['actual_size_kb']:.1f} KB (Limit: {spec['max_size_kb']} KB)")
            else:
                print(f"  ⚠️  File size: {results['actual_size_kb']:.1f} KB (Limit: {spec['max_size_kb']} KB)")
                print(f"      Warning: Exceeds size limit")
        
        print("\n" + "="*70)
        if all_passed:
            print("✅ ALL VALIDATIONS PASSED")
        else:
            print("❌ SOME VALIDATIONS FAILED")
        print("="*70 + "\n")
        
        return all_passed


def main():
    """Main validation function."""
    validator = AssetValidator()
    
    # Find base name from existing files
    output_dir = "output/secondary_assets"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        if files:
            # Extract base name from first file
            base_name = files[0].split('_Image_')[0].split('_Logo_')[0]
            validator.validate_all_assets(base_name)
        else:
            print("❌ No files found in output directory")
    else:
        print("❌ Output directory not found")


if __name__ == "__main__":
    main()