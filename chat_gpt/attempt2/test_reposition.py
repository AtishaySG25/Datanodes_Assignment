from preprocess import parse_psd
from object_detection import classify_layers
from importance_map import rank_objects
from reposition import compute_layout

scene = parse_psd("D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd")
labels = classify_layers(scene["layers"], scene["canvas_size"])
ranked = rank_objects(scene["layers"], labels)

layout = compute_layout(
    canvas_size=scene["canvas_size"],
    target_size=(970, 90),
    layers=scene["layers"],
    ranked_objects=ranked
)

for l in layout:
    print(l)
