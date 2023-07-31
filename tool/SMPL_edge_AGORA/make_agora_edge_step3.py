import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
# putting your path ex) "/home/user/SEFD/dataset"
base_dir = "putting your path/AGORA/SMPL_edge_55"
target_dir = "putting your path/AGORA/SMPL_overlap_edge"

edge_list=os.listdir(base_dir)

print(len(edge_list))

name=""
img_list=[]
for n,i in enumerate(tqdm(sorted(edge_list))):
    if n==0:
        name=i.split("1280x720")[0]
        img=cv2.imread(osp.join(base_dir,i),cv2.IMREAD_GRAYSCALE)
        img_list.append(img)
    else:
        if name==i.split("1280x720")[0]:
            img=cv2.imread(osp.join(base_dir,i),cv2.IMREAD_GRAYSCALE)
            img_list.append(img)
        else:
            zeros_img=np.zeros((720,1280))
            for imgs in img_list:
                zeros_img+=imgs/255.
            zeros_img=np.minimum(1,zeros_img)
            cv2.imwrite(osp.join(target_dir,name+"1280x720.jpg"),zeros_img*255)
            name=i.split("1280x720")[0]
            img_list=[]
            img=cv2.imread(osp.join(base_dir,i),cv2.IMREAD_GRAYSCALE)
            img_list.append(img)