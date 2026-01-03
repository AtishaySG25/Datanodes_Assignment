# psd_loader.py
from psd_tools import PSDImage
from PIL import Image

def load_psd_as_image(psd_path):
    """
    Loads a PSD file and returns a high-quality composited PIL Image (RGBA)
    """
    psd = PSDImage.open(psd_path)
    composite = psd.composite()
    # Ensure RGBA for transparency handling
    if composite.mode != 'RGBA':
        composite = composite.convert('RGBA')
    return composite

def get_psd_info(psd_path):
    psd = PSDImage.open(psd_path)
    return {
        'size': psd.size,
        'layers': [layer.name for layer in psd if layer.is_visible()]
    }