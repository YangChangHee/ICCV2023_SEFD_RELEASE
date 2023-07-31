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


# Putting Path Data Directory(MPII)
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

def mpii_smpl_Edge(data_path):
    data_path=osp.join(data_path,"MPII")
    ced=canny_edge()

    if not os.path.exists(data_path+'/SMPL_all_edge'):
        os.makedirs(data_path+'/SMPL_all_edge')
        print("make of directory : "+data_path+'/SMPL_all_edge')
    if not os.path.exists(data_path+'/SMPL_overlap_edge'):
        os.makedirs(data_path+'/SMPL_overlap_edge')
        print("make of directory : "+data_path+'/SMPL_overlap_edge')


    with open(osp.join(data_path,"annotations", 'MPII_train_SMPL_NeuralAnnot.json')) as f:
        smpl_params = json.load(f)

    db = COCO(osp.join(data_path,"annotations", 'train.json'))

    for iid in db.imgs.keys():

        aids = db.getAnnIds([iid])
        image_file_name=db.imgs[iid]['file_name']
        list_name=image_file_name.split('/')
        h1,w2=0,0
        list_mesh=[]
        list_rendered_edge=[]
        list_camera_parma=[]
        #if not os.path.exists(data_path+'/SMPL_multi/'+'SMPL_multi_'+list_name[1]):
        for aid in aids:
            ann=db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            h1,w2=img['height'], img['width']
            threshold=ann['bbox'][2]*ann['bbox'][3]
            img=np.ones((h1,w2,3))*255
            if str(aid) in smpl_params:
                person_smpl=smpl_params[str(aid)]
                smpl_mesh_coord, _ =get_smpl_coord(person_smpl['smpl_param'],benchmark='mscoco')
                list_camera_parma.append(person_smpl['cam_param'])
                mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
                list_mesh.append(mesh_cam[0].cpu().numpy())
                

        length=len(list_mesh)
        if length >=1:
            for i in range(length-1):
                x=np.array(list_camera_parma[i]['princpt'][0])- np.array(list_camera_parma[-1]['princpt'][0])
                y=np.array(list_camera_parma[i]['princpt'][1])- np.array(list_camera_parma[-1]['princpt'][1])

                tt=np.array(list_camera_parma[i]['focal'])/np.array(list_camera_parma[-1]['focal'])
                
                list_mesh[i][:,2]=list_mesh[i][:,2]/tt[0]
                ttx=list_mesh[i][:,2]/np.array(list_camera_parma[-1]['focal'])[0] * x
                tty=list_mesh[i][:,2]/np.array(list_camera_parma[-1]['focal'])[1] * y
                list_mesh[i][:,0]=list_mesh[i][:,0]+ttx
                list_mesh[i][:,1]=list_mesh[i][:,1]+tty

            for n,i in enumerate(list_mesh):
                img=np.ones((h1,w2,3))*255
                rendered_img=render_mesh(img,i,face,list_camera_parma[-1])
                
                ts=torch.FloatTensor(rendered_img)/255.

                
                ts=ts[None,:,:,:].cuda()
                ts=ts.transpose(1,3).transpose(2,3)
                tt=ced(ts)
                list_rendered_edge.append(tt)

            img2=np.ones((h1,w2,3))*255
            result=mscoco_multi_render_mesh(img2,list_mesh,face,list_camera_parma)
            ts1=torch.FloatTensor(result)/255.
            ts1=ts1[None,:,:,:].cuda()
            ts1=ts1.transpose(1,3).transpose(2,3)
            tt1=ced(ts1)
            test=np.zeros((h1,w2))
            for j in list_rendered_edge:
                test+=j[1][0][0].cpu().numpy()
            result1=np.minimum(result1,1)

            cv2.imwrite(data_path+'/SMPL_all_edge/'+'SMPL_all_edge_'+list_name[1],tt1[1][0][0].cpu().numpy()*255)
            cv2.imwrite(data_path+'/SMPL_overlap_edge/'+'SMPL_overlap_edge_'+list_name[1],result1*255)
        else:
            pass


if __name__ =="__main__":
    mpii_smpl_Edge(data_path)