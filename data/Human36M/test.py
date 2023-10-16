import json

with open("/database/changhee/Human36M/densepose_annot/human_parsing_bbox.json","r") as f:
    bbox_list=json.load(f)

print(sorted(bbox_list.keys()))