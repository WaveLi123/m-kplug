from PIL import Image
import os
import sys

input_path = sys.argv[1]
output_path = input_path + "_rgb"
if not os.path.exists(output_path):
    os.mkdir(output_path)

for idx, img_name in enumerate(os.listdir(input_path)):
    img = Image.open(os.path.join(input_path, img_name), 'r')

    if(img.mode!='RGB'):
        print(idx, img_name, "is not rgb image")
        img = img.convert("RGB")
        img.save(os.path.join(output_path, img_name))
