import sys
# putting your path ex) "/home/user/SEFD/dataset"
sys.path.append("putting your path/SEFD/common/")
sys.path.append("putting your path/SEFD/data/")
sys.path.append("putting your path/SEFD/main/")
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
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle
import kornia as K
from torch import Tensor
from kornia.color import bgr_to_rgb,rgb_to_grayscale
from kornia.filters import canny, gaussian_blur2d, motion_blur
from kornia.morphology import dilation, closing, erosion
import cv2
import sys
import numpy as np
import torch.nn as nn
from tqdm import tqdm


# Putting Path Data Directory(MuCo)
data_path = 'putting your path'


smpl = SMPL()
face = smpl.face
joint_regressor = smpl.joint_regressor
vertex_num = smpl.vertex_num
joint_num = smpl.joint_num
joints_name = smpl.joints_name
skeleton = smpl.skeleton
root_joint_idx = smpl.root_joint_idx
face_kps_vertex = smpl.face_kps_vertex


muco_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
muco_root_joint_idx = muco_joints_name.index('Pelvis')


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
    smpl_joint_coord = np.dot(joint_regressor, smpl_mesh_coord)

    return smpl_mesh_coord, smpl_joint_coord

class canny_edge(nn.Module):
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter
        self.kernel=torch.ones(cfg.dilation[0]).cuda()
        self.kernel1=torch.ones(cfg.dilation[1]).cuda()
        self.kernel2=torch.ones(cfg.dilation[2]).cuda()
        self.kernel3=torch.ones(cfg.dilation[3]).cuda()
        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)
        self.chshuffle=RandomChannelShuffle(p=0.75)
    # gradient
    #@torch.no_grad() # No_grad
    def forward(self, x:Tensor) -> Tensor:
        x_out5=self.chshuffle(x)
        x=bgr_to_rgb(x)
        x=rgb_to_grayscale(x)
        x_out=canny(x)
#        x_out_t=x_out[0].cuda()
        x_out1=dilation(x_out[0],kernel=self.kernel)
        x_out2=dilation(x_out[0],kernel=self.kernel1)
        x_out3=dilation(x_out[0],kernel=self.kernel2)
        x_out4=dilation(x_out[0],kernel=self.kernel3)
        if self._apply_color_jitter:
            print(self._apply_color_jitter)
            x_out5 = self.jitter(x_out5)
        
        return x_out[0],x_out1,x_out2,x_out3,x_out4, x_out5

def make_muco_smpl_Edge(data_path):
    data_path=osp.join(data_path,"MuCo")
    ced=canny_edge()
    if not os.path.exists(data_path+'/SMPL_all_edge'):
        os.makedirs(data_path+'/SMPL_all_edge')
        print("make of directory : "+data_path+'/SMPL_all_edge')
    if not os.path.exists(data_path+'/SMPL_overlap_edge'):
        os.makedirs(data_path+'/SMPL_overlap_edge')
        print("make of directory : "+data_path+'/SMPL_overlap_edge')
    
    smpl_param_path = osp.join(data_path, 'data', 'smpl_param.json')
    annot_path = osp.join(data_path, 'data', 'MuCo-3DHP.json')
    db = COCO(annot_path)
    with open(smpl_param_path) as f:
        smpl_params = json.load(f)
    

    for iid in db.imgs.keys():
        img = db.imgs[iid]
        img_id = img["id"]
        img_width, img_height = img['width'], img['height']
        imgname = img['file_name']
        list_split=imgname.split('/')
        focal = img["f"]
        princpt = img["c"]
        cam_param = {'focal': focal, 'princpt': princpt}
        #print(cam_param)
        if not os.path.exists(data_path+'/SMPL_overlap_edge/'+list_split[0]+'/'+list_split[1]):
            os.makedirs(data_path+'/SMPL_overlap_edge/'+list_split[0]+'/'+list_split[1])
            print("make of directory : "+data_path+'/SMPL_overlap_edge/'+list_split[0]+'/'+list_split[1])
        if not os.path.exists(data_path+'/SMPL_all_edge/'+list_split[0]+'/'+list_split[1]):
            os.makedirs(data_path+'/SMPL_all_edge/'+list_split[0]+'/'+list_split[1])
            print("make of directory : "+data_path+'/SMPL_all_edge/'+list_split[0]+'/'+list_split[1])
        ann_ids = db.getAnnIds(img_id)
        anns = db.loadAnns(ann_ids)
        root_depths = [ann['keypoints_cam'][muco_root_joint_idx][2] for ann in anns]
        closest_pid = root_depths.index(min(root_depths))
        pid_list = [closest_pid]
        current_path=list_split[2]
        list_mesh=[]
        list_rendered_edge=[]
        for pid in pid_list:
            try:
                smpl_param = smpl_params[str(ann_ids[pid])]
                smpl_mesh_coord, _=get_smpl_coord(smpl_param,benchmark='mscoco')
                img=np.ones((img_height,img_width,3))*255
                mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
                list_mesh.append(mesh_cam[0].cpu().numpy())
                rendered_img=render_mesh(img,list_mesh[-1],face,cam_param)
                ts=torch.FloatTensor(rendered_img)/255.
                ts=ts[None,:,:,:].cuda()
                ts=ts.transpose(1,3).transpose(2,3)
                tt=ced(ts)
                list_rendered_edge.append(tt)
            except:
                pass
        img1=np.ones((img_height,img_width,3))*255
        rendered_img2 = multi_render_mesh(img1, list_mesh, face, cam_param)
        ts1=torch.FloatTensor(rendered_img2)/255.
        ts1=ts1[None,:,:,:].cuda()
        ts1=ts1.transpose(1,3).transpose(2,3)
        tt1=ced(ts1)
        test=np.zeros((img_height,img_width))
        for j in list_rendered_edge:
            test+=j[2][0][0].cpu().numpy()
        result=np.minimum(result,1)
        cv2.imwrite(data_path+'/SMPL_overlap_edge/'+list_split[0]+'/'+list_split[1]+'/'+'SMPL_overlap_edge_'+current_path,result*255)
        cv2.imwrite(data_path+'/SMPL_all_edge/'+list_split[0]+'/'+list_split[1]+'/'+'SMPL_all_edge_'+current_path,tt1[2][0][0].cpu().numpy()*255)



if __name__ =="__main__":
    make_muco_smpl_Edge(data_path)