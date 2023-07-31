import sys
sys.path.append("putting your path/3DCrowdNet_RELEASE/common/")
sys.path.append("putting your path/3DCrowdNet_RELEASE/data/")
sys.path.append("putting your path/3DCrowdNet_RELEASE/main/")
import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from config import cfg
from utils.renderer import Renderer
from utils.smpl import SMPL
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db, denorm_joints, convert_crop_cam_to_orig_img
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, vis_bbox, render_mesh, multi_render_mesh, new_multi_render_mesh, mscoco_multi_render_mesh

from utils.transforms import rot6d_to_axis_angle
import cv2
import torch.nn as nn

import time
from tqdm import tqdm

smpl = SMPL()
face = smpl.face
joint_regressor = smpl.joint_regressor
vertex_num = smpl.vertex_num
joint_num = smpl.joint_num
joints_name = smpl.joints_name
skeleton = smpl.skeleton
root_joint_idx = smpl.root_joint_idx
face_kps_vertex = smpl.face_kps_vertex

def get_smpl_coord(smpl_param,benchmark='noraml'):
    if benchmark =='normal':
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
    if benchmark =='mscoco':
        pose, shape, trans= smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
    smpl_pose = torch.FloatTensor(pose).view(1,-1); smpl_shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
    smpl_trans = torch.FloatTensor(trans).view(-1,3) # translation vector from smpl coordinate to 3dpw camera coordinate

    # TEMP
    # gender = 'neutral'
    # get mesh and joint coordinates
    if benchmark == 'normal':
        smpl_mesh_coord, smpl_joint_coord = smpl.layer[gender](smpl_pose, smpl_shape, smpl_trans)
    elif benchmark == 'mscoco':
        smpl_mesh_coord, smpl_joint_coord = smpl.layer['neutral'](smpl_pose, smpl_shape, smpl_trans)

    # incorporate face keypoints
    smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3);
    # smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
    # smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
    # smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))
    smpl_joint_coord = np.dot(joint_regressor, smpl_mesh_coord)

    return smpl_mesh_coord, smpl_joint_coord

def make_muco_smpl(dir_list):
    data_path="putting your path/MuCo"
    if not os.path.exists(data_path+'/SMPL'):
        os.makedirs(data_path+'/SMPL')
        print("make of directory : "+data_path+'/SMPL')

    smpl_param_path = osp.join(data_path, 'data', 'smpl_param.json')
    annot_path = osp.join(data_path, 'data', 'MuCo-3DHP.json')
    db = COCO(annot_path)
    with open(smpl_param_path) as f:
        smpl_params = json.load(f)

    for iid in tqdm(db.imgs.keys()):
        img = db.imgs[len(db.imgs.keys())-1-iid]
        img_id = img["id"]
        img_width, img_height = img['width'], img['height']
        imgname = img['file_name']
        list_split=imgname.split('/')
        focal = img["f"]
        princpt = img["c"]
        cam_param = {'focal': focal, 'princpt': princpt}
        if int(list_split[1])>=dir_list[0] and int(list_split[1]) <=dir_list[1]:
            if not os.path.exists(data_path+'/SMPL/'+list_split[0]+'/'+list_split[1]):
                os.makedirs(data_path+'/SMPL/'+list_split[0]+'/'+list_split[1])
                print("make of directory : "+data_path+'/SMPL/'+list_split[0]+'/'+list_split[1])
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)
            current_path=list_split[2]
            for pid in ann_ids:
                try:
                    smpl_param = smpl_params[str(pid)]
                    smpl_mesh_coord, _=get_smpl_coord(smpl_param,benchmark='mscoco')
                    img=np.ones((img_height,img_width,3))*255
                    mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
                    rendered_img=render_mesh(img,mesh_cam[0].cpu().numpy(),face,cam_param)
                    cv2.imwrite(data_path+'/SMPL/'+list_split[0]+'/'+list_split[1]+'/'+current_path[:-4]+'_{}.jpg'.format(pid),rendered_img)
                except:
                    continue


make_muco_smpl([0,10])