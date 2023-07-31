import json

with open("agora_smpl_edge_2d.json") as f:
    data=json.load(f)
print(data['ag_trainset_3dpeople_bfh_archviz_5_10_cam00_00000_1280x720.png'][0]['0'])