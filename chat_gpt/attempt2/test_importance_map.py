from preprocess import parse_psd
from object_detection import classify_layers
from importance_map import rank_objects

scene = parse_psd("D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd")
labels = classify_layers(scene["layers"], scene["canvas_size"])
ranked = rank_objects(scene["layers"], labels)

for r in ranked:
    print(r)
