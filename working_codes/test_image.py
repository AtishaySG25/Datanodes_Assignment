from psd_tools import PSDImage

psd_path = "C:/Users/ATISHAY SG/Downloads/2912061.psd"

psd = PSDImage.open(psd_path)

width, height = psd.width, psd.height
print(f"PSD dimensions: {width} x {height}")

