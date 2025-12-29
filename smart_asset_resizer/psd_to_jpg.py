from psd_tools import PSDImage
from PIL import Image
import os

def convert_psd_to_jpeg(psd_path, jpeg_path, quality=85):
    """
    Converts a PSD file to a JPEG file.

    Args:
        psd_path (str): The path to the input .psd file.
        jpeg_path (str): The path for the output .jpeg file.
        quality (int): The quality of the output JPEG (1-95, default 85).
    """
    try:
        # Open the PSD file
        psd = PSDImage.open(psd_path)
        
        # Composite all layers into a single Pillow Image object
        # `force=True` handles potential issues with complex PSD files
        merged_image = psd.composite(force=True)

        # Ensure the image is in RGB mode, as JPEG does not support CMYK or RGBA directly
        if merged_image.mode in ('RGBA', 'CMYK'):
            if merged_image.mode == 'CMYK':
                # Convert CMYK to RGB (color shifts may occur without proper ICC profile handling)
                merged_image = merged_image.convert('RGB')
            elif merged_image.mode == 'RGBA':
                # Convert RGBA to RGB by creating a white background
                background = Image.new("RGB", merged_image.size, (255, 255, 255))
                background.paste(merged_image, mask=merged_image.split()[3]) # Use alpha channel as mask
                merged_image = background

        # Save the image as JPEG
        merged_image.save(jpeg_path, 'JPEG', quality=quality)
        print(f"Successfully converted '{psd_path}' to '{jpeg_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{psd_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example Usage:
input_file = 'smart_asset_resizer/input/Axis_Multicap_fund.psd'
output_file = 'smart_asset_resizer/input/Axis_Multicap_fund.jpg'
convert_psd_to_jpeg(input_file, output_file)
