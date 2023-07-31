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


# Putting Path Data Directory(3DPW)
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

def make_3dpw_smpl_Edge(data_path):
    ced=canny_edge()
    if not os.path.exists(data_path+'/3dpw/SMPL_edge/'):
        os.makedirs(data_path+'/3dpw/SMPL_edge/')
        print("make of directory : "+data_path+'/3dpw/SMPL_edge/')
    db = COCO(osp.join(data_path,'3dpw','data', '3DPW_latest_' + 'test' + '.json'))
    person_num=0
    current_path=''
    currnet_dir=''
    list_mesh=[]
    list_rendered_edge=[]
    before_w,before_h=0,0
    for i in range(len(db.anns)):
        #if db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] == "downtown_runForBus_01" or db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] == "downtown_walkBridge_01" or db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] == "downtown_walkUphill_00" or db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] ==  "flat_guitar_01" :
        if False: # test
            pass
        else:
            if currnet_dir != db.loadImgs(db.anns[i]['image_id'])[0]['sequence']:
                currnet_dir=db.loadImgs(db.anns[i]['image_id'])[0]['sequence']
                if not os.path.exists(data_path+'/3dpw/SMPL_all_edge/'+currnet_dir):
                    os.makedirs(data_path+'/3dpw/SMPL_all_edge/'+currnet_dir)
                    print("make of directory : " +data_path+'/3dpw/SMPL_all_edge/'+currnet_dir)
                if not os.path.exists(data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir):
                    os.makedirs(data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir)
                    print("make of directory : " +data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir)
            mesh,w,h=db.anns[i]['smpl_param'],db.loadImgs(db.anns[i]['image_id'])[0]['width'], db.loadImgs(db.anns[i]['image_id'])[0]['height']
            smpl_mesh_coord, _=get_smpl_coord(mesh,benchmark='normal')
            save_obj(smpl_mesh_coord,face,'test.obj_{}'.format(i))
            if i==1:
                import sys
                sys.exit()
            mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
            w,h=db.loadImgs(db.anns[i]['image_id'])[0]['width'], db.loadImgs(db.anns[i]['image_id'])[0]['height']
            img=np.ones((h,w,3))*255
            threshold=db.anns[i]['bbox'][2]*db.anns[i]['bbox'][3]
            #print(threshold)
            
            if current_path !=db.loadImgs(db.anns[i]['image_id'])[0]['file_name']:
                if list_mesh !=[]:
                    current_path=db.loadImgs(db.anns[i]['image_id'])[0]['file_name']

                    img1=np.ones((before_h,before_w,3))*255
                    rendered_img2 = multi_render_mesh(img1, list_mesh, face, db.loadImgs(db.anns[i]['image_id'])[0]['cam_param'])
                    ts1=torch.FloatTensor(rendered_img2)/255.
                    ts1=ts1[None,:,:,:].cuda()
                    ts1=ts1.transpose(1,3).transpose(2,3)

                    tt1=ced(ts1)
                    test=np.zeros((before_h,before_w))
                    if threshold<=256:
                        for j in list_rendered_edge:
                            test+=j[0][0][0].cpu().numpy()
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir+'/'+'SMPL_overlap_edge_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_all_edge/'+currnet_dir+'/'+'SMPL_all_edge_'+current_path,tt1[0][0][0].cpu().numpy()*255)
                    elif threshold<=4096 and threshold>256:
                        for j in list_rendered_edge:
                            test+=j[1][0][0].cpu().numpy()
                        result=np.maximum(test-tt1[1][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir+'/'+'SMPL_overlap_edge_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_all_edge/'+currnet_dir+'/'+'SMPL_all_edge_'+current_path,tt1[1][0][0].cpu().numpy()*255)
                    elif threshold<=65536 and threshold>4096:
                        for j in list_rendered_edge:
                            test+=j[2][0][0].cpu().numpy()
                        result=np.maximum(test-tt1[2][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir+'/'+'SMPL_overlap_edge_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_all_edge/'+currnet_dir+'/'+'SMPL_all_edge_'+current_path,tt1[2][0][0].cpu().numpy()*255)
                    elif threshold>65536:
                        for j in list_rendered_edge:
                            test+=j[3][0][0].cpu().numpy()
                        result=np.maximum(test-tt1[3][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_overlap_edge/'+currnet_dir+'/'+'SMPL_overlap_edge_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_all_edge/'+currnet_dir+'/'+'SMPL_all_edge_'+current_path,tt1[3][0][0].cpu().numpy()*255)
                    
                    
                    list_mesh=[]
                    list_rendered_edge=[]
                    #print("go")
                    list_mesh.append(mesh_cam[0].cpu().numpy())
                    
                    rendered_img=render_mesh(img,list_mesh[-1],face,db.loadImgs(db.anns[i]['image_id'])[0]['cam_param'])
                    ts=torch.FloatTensor(rendered_img)/255.
                    ts=ts[None,:,:,:].cuda()
                    ts=ts.transpose(1,3).transpose(2,3)
                    tt=ced(ts)
                    list_rendered_edge.append(tt)
                    before_w,before_h=w,h
                else:
                    list_mesh.append(mesh_cam[0].cpu().numpy())
                    
                    rendered_img=render_mesh(img,list_mesh[-1],face,db.loadImgs(db.anns[i]['image_id'])[0]['cam_param'])
                    ts=torch.FloatTensor(rendered_img)/255.
                    ts=ts[None,:,:,:].cuda()
                    ts=ts.transpose(1,3).transpose(2,3)
                    tt=ced(ts)
                    list_rendered_edge.append(tt)
                    before_w,before_h=w,h
            else:
                list_mesh.append(mesh_cam[0].cpu().numpy())
                
                rendered_img=render_mesh(img,list_mesh[-1],face,db.loadImgs(db.anns[i]['image_id'])[0]['cam_param'])
                ts=torch.FloatTensor(rendered_img)/255.
                ts=ts[None,:,:,:].cuda()
                ts=ts.transpose(1,3).transpose(2,3)
                tt=ced(ts)
                list_rendered_edge.append(tt)
                before_w,before_h=w,h
            

if __name__ =="__main__":
    make_3dpw_smpl_Edge(data_path)