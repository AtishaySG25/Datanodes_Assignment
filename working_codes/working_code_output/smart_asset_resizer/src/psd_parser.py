from psd_tools import PSDImage
from src.objects import DesignObject


def classify_layer(layer, psd_size):
    if layer.kind == "type":
        return "text", 10
    if layer.kind == "smartobject":
        return "logo", 8
    if layer.size == psd_size:
        return "background", 0
    return "graphic", 5


def extract_layers_recursive(layer, psd_size, elements):
    """
    Recursively traverse PSD groups and layers
    """
    if not layer.visible:
        return

    if layer.is_group():
        for child in layer:
            extract_layers_recursive(child, psd_size, elements)
        return

    img = layer.composite()
    if img is None:
        return

    obj_type, priority = classify_layer(layer, psd_size)
    elements.append((priority, DesignObject(img.convert("RGBA"), obj_type)))


def extract_design_objects(psd_path):
    psd = PSDImage.open(psd_path)
    elements = []

    for layer in psd:
        extract_layers_recursive(layer, psd.size, elements)

    # Sort by importance (highest first)
    elements.sort(key=lambda x: x[0], reverse=True)

    # Return only DesignObjects
    return [obj for _, obj in elements]
