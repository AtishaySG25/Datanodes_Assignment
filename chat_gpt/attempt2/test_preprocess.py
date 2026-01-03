from preprocess import parse_psd

data = parse_psd("D:/Datanodes_Assignment/input/Axis_Multicap_fund.psd")

print("Canvas:", data["canvas_size"])
print("Total layers:", len(data["layers"]))

for l in data["layers"]:
    print(l["id"], l["type"], l["bbox"])
