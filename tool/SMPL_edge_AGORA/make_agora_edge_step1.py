import sys
# putting your path ex) "/home/user/SEFD/dataset"
sys.path.append("putting your path/SEFD/common/")
sys.path.append("putting your path/SEFD/data/")
sys.path.append("putting your path/SEFD/main/")
import os
import os.path as osp

import numpy as np
from utils.vis import render_mesh
import time
from tqdm import tqdm
import cv2
import json
from utils.smpl import SMPL

smpl = SMPL()
face = smpl.face

# Putting Path Data Directory (AGORA)
data_path = 'putting your path'

def AGORA_smpl_edge(data_path):
    json_data={}
    data_path=osp.join(data_path,"AGORA")
    if not os.path.exists(data_path+'/SMPL_edge'):
        os.makedirs(data_path+'/SMPL_edge')
        print("make of directory : "+data_path+'/SMPL_edge')
    test=np.load("putting your path/AGORA/annots_train.npz", allow_pickle=True)['annots'][()]
    file_list=list(test.keys())
    for i in tqdm(sorted(file_list)):
        json_data[i]=[]
        agora=np.load(osp.join(data_path,"image_vertex_train",i[:-4]+".npz"), allow_pickle=True)['verts']
        #print(agora.shape)
        json_data1={}
        for n,j in enumerate(agora):
            prict=test[i][n]['camMats'][0][0]
            camera_center = np.array([1280 / 2., 720 / 2.])
            focal=np.array([prict,prict])
            cam_param = {'focal': focal, 'princpt': camera_center}
            kp2d=test[i][n]['kp2d']
            img=np.ones((720,1280,3))*255
            rendered_img=render_mesh(img,j,face,cam_param)
            cv2.imwrite(osp.join(data_path,"SMPL_edge","{}_{}.jpg".format(i,n)),rendered_img)
            json_data1[n]=kp2d.tolist()
            json_data[i].append(json_data1)

    with open("agora_smpl_edge_2d.json",'w') as make_file:
        json.dump(json_data,make_file)


if __name__ =="__main__":
    AGORA_smpl_edge(data_path)