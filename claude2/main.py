"""
main.py

PSD Banner Generator
Transforms a 1080x1080 master PSD into multiple banner formats
while preserving brand guidelines, colors, and layer hierarchy.

Usage:
    python main.py --input input/master_assets/sample_master.psd --output output/
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from src.psd_parser import PSDParser
from src.layer_classifier import LayerClassifier
from src.layout_engine import LayoutEngine
from src.compositor import BrandSafeCompositor
from src.validator import BrandValidator
from src.utils.logger import setup_logger

# Output specifications
OUTPUT_SPECS = [
    {"name": "leaderboard_large", "width": 970, "height": 90},
    {"name": "leaderboard_medium", "width": 728, "height": 90},
    {"name": "skyscraper", "width": 160, "height": 600},
    {"name": "banner", "width": 468, "height": 60},
    {"name": "small_square", "width": 200, "height": 200},
    {"name": "medium_rectangle", "width": 300, "height": 250},
]


class PSDBannerGenerator:
    """
    Main orchestrator for PSD to banner transformation
    """
    
    def __init__(self, input_path: str, output_dir: str, debug: bool = False):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        # Setup logger
        self.logger = setup_logger(debug=debug)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = None
        self.classifier = None
        self.layout_engine = None
        self.compositor = None
        self.validator = None
        
    def process(self) -> Dict[str, str]:
        """
        Main processing pipeline
        
        Returns:
            Dictionary mapping format names to output file paths
        """
        self.logger.info(f"Starting PSD processing: {self.input_path}")
        
        # Step 1: Parse PSD
        self.logger.info("Step 1/6: Parsing PSD file...")
        self.parser = PSDParser(str(self.input_path))
        psd_data = self.parser.parse()
        self.logger.info(f"Parsed {len(psd_data['layers'])} layers")
        
        # Step 2: Classify layers
        self.logger.info("Step 2/6: Classifying layers...")
        self.classifier = LayerClassifier(psd_data)
        classified_layers = self.classifier.classify()
        self._log_classification(classified_layers)
        
        # Step 3: Extract brand guidelines
        self.logger.info("Step 3/6: Extracting brand guidelines...")
        brand_guidelines = self.classifier.extract_brand_guidelines()
        self._log_brand_guidelines(brand_guidelines)
        
        # Step 4: Generate each output format
        self.logger.info("Step 4/6: Generating banner formats...")
        output_files = {}
        
        for spec in OUTPUT_SPECS:
            try:
                self.logger.info(f"  Generating {spec['name']} ({spec['width']}x{spec['height']})...")
                
                # Layout calculation
                self.layout_engine = LayoutEngine(
                    classified_layers,
                    brand_guidelines,
                    target_size=(spec['width'], spec['height'])
                )
                layout = self.layout_engine.calculate_layout()
                
                # Composite image
                self.compositor = BrandSafeCompositor(
                    psd_data,
                    brand_guidelines
                )
                output_image = self.compositor.composite(layout)
                
                # Validate output
                self.validator = BrandValidator(brand_guidelines)
                validation_result = self.validator.validate(output_image, layout)
                
                if not validation_result['passed']:
                    self.logger.warning(f"  Validation warnings for {spec['name']}: {validation_result['warnings']}")
                
                # Save output
                output_path = self.output_dir / f"{spec['name']}_{spec['width']}x{spec['height']}.png"
                output_image.save(output_path, 'PNG', icc_profile=psd_data.get('icc_profile'))
                output_files[spec['name']] = str(output_path)
                
                self.logger.info(f"  ✓ Saved: {output_path}")
                
            except Exception as e:
                self.logger.error(f"  ✗ Failed to generate {spec['name']}: {str(e)}")
                if self.debug:
                    raise
        
        self.logger.info(f"Step 5/6: Generation complete. {len(output_files)}/{len(OUTPUT_SPECS)} successful")
        
        # Step 6: Generate report
        self.logger.info("Step 6/6: Generating summary report...")
        self._generate_report(output_files, brand_guidelines)
        
        return output_files
    
    def _log_classification(self, classified_layers: Dict):
        """Log layer classification results"""
        for category, layers in classified_layers.items():
            if layers:
                layer_names = [l['name'] for l in layers]
                self.logger.info(f"  {category.upper()}: {', '.join(layer_names)}")
    
    def _log_brand_guidelines(self, guidelines: Dict):
        """Log extracted brand guidelines"""
        self.logger.info(f"  Brand colors: {len(guidelines['colors'])} detected")
        self.logger.info(f"  Logo size: {guidelines['logo_size']}")
        self.logger.info(f"  Typography: {len(guidelines['fonts'])} font families")
    
    def _generate_report(self, output_files: Dict, brand_guidelines: Dict):
        """Generate HTML summary report"""
        report_path = self.output_dir / "generation_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PSD Banner Generation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
                h1 {{ color: #333; }}
                .banner {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; background: #fafafa; }}
                .banner img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
                .brand-info {{ background: #e8f4f8; padding: 15px; border-radius: 4px; margin: 20px 0; }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>PSD Banner Generation Report</h1>
                <p><strong>Source:</strong> {self.input_path.name}</p>
                <p><strong>Generated:</strong> {len(output_files)} of {len(OUTPUT_SPECS)} formats</p>
                
                <div class="brand-info">
                    <h2>Brand Guidelines Applied</h2>
                    <p><strong>Colors:</strong> {', '.join([f'RGB{c}' for c in brand_guidelines['colors'][:5]])}</p>
                    <p><strong>Logo Size:</strong> {brand_guidelines['logo_size']}</p>
                </div>
                
                <h2>Generated Banners</h2>
        """
        
        for spec in OUTPUT_SPECS:
            if spec['name'] in output_files:
                rel_path = Path(output_files[spec['name']]).name
                html_content += f"""
                <div class="banner">
                    <h3 class="success">✓ {spec['name']} ({spec['width']}x{spec['height']})</h3>
                    <img src="{rel_path}" alt="{spec['name']}">
                </div>
                """
            else:
                html_content += f"""
                <div class="banner">
                    <h3 class="warning">✗ {spec['name']} ({spec['width']}x{spec['height']}) - Generation Failed</h3>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Transform PSD master asset into multiple banner formats'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input PSD file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for generated banners'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.lower().endswith('.psd'):
        print(f"Error: Input must be a PSD file")
        sys.exit(1)
    
    # Run generator
    generator = PSDBannerGenerator(
        input_path=args.input,
        output_dir=args.output,
        debug=args.debug
    )
    
    try:
        output_files = generator.process()
        print(f"\n✓ Successfully generated {len(output_files)} banner formats")
        print(f"Output directory: {args.output}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())