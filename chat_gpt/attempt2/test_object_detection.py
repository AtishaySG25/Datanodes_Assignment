from preprocess import parse_psd
from object_detection import classify_layers

scene = parse_psd("D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd")
labels = classify_layers(scene["layers"], scene["canvas_size"])

for l in labels:
    print(l)
