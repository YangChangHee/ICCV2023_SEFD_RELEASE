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
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, vis_bbox, render_mesh, multi_render_mesh, new_multi_render_mesh, mscoco_multi_render_mesh, h36m_render_mesh

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


# Putting Path Data Directory(Human36M)
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


def h36m_get_smpl_coord( smpl_param, cam_param):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(-1,3); smpl_shape = torch.FloatTensor(shape).view(1,-1); 
        R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3,3), np.array(cam_param['t'], dtype=np.float32).reshape(3)
        smpl_trans = np.array(trans, dtype=np.float32).reshape(3)
        smpl_trans = np.dot(R, smpl_trans[:,None]).reshape(1,3) + t.reshape(1,3)/1000
        smpl_trans = torch.FloatTensor(smpl_trans).view(1,3) 
        
        root_pose = smpl_pose[root_joint_idx,:].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R,root_pose))
        smpl_pose[root_joint_idx] = torch.from_numpy(root_pose).view(3)
        
        smpl_pose = smpl_pose.view(1,-1)
       
        smpl_mesh_coord, smpl_joint_coord = smpl.layer['neutral'](smpl_pose, smpl_shape,smpl_trans)


        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3);
        smpl_joint_coord = np.dot(joint_regressor, smpl_mesh_coord)


        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy()


def make_h36m_smpl_Edge(data_path):
    sampling_ratio=5
    ced=canny_edge()
    annot_path = osp.join(data_path, 'Human36M', 'annotations')
    img_dir = osp.join(data_path,  'Human36M', 'images')
    data_path = osp.join(data_path,'Human36M')
    if not os.path.exists(data_path+'/SMPL_edge/'):
        os.makedirs(data_path+'/SMPL_edge/')
        print("make of directory : "+data_path+'/SMPL_edge/')
    subject_list = [8]
    db = COCO()
    cameras = {}
    joints = {}
    smpl_params = {}
    for subject in subject_list:
        # data load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
            annot = json.load(f)
        if len(db.dataset) == 0:
            for k,v in annot.items():
                db.dataset[k] = v
        else:
            for k,v in annot.items():
                db.dataset[k] += v
        # camera load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
            cameras[str(subject)] = json.load(f)
        # joint coordinate load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
            joints[str(subject)] = json.load(f)
        # smpl parameter load
        with open(osp.join(annot_path, 'Human36M_subject' + str(subject) + '_smpl_param.json'),'r') as f:
            smpl_params[str(subject)] = json.load(f)
    db.createIndex()
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']
        img = db.loadImgs(image_id)[0]
        list_split=img['file_name'].split('/')
        combine_list=osp.join(list_split[0],list_split[1])
        img_shape = (img['height'], img['width'])
        subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
        if frame_idx % sampling_ratio != 0:
            continue
        try:
            smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
        except KeyError:
            smpl_param = None
        #print(smpl_param)
            
        cam_idx = img['cam_idx']
        if not os.path.exists(data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx)):
            os.makedirs(data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx))
            print("make of directory : "+data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx))
        threshold=np.array(ann['bbox'])[2]*np.array(ann['bbox'])[3]
        cam_param = cameras[str(subject)][str(cam_idx)]
        R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
        cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
        try:
            smpl_mesh_coord,_,_,_=h36m_get_smpl_coord(smpl_param,cam_param)
            t_img=np.ones((img['height'], img['width'],3))
            vjs_tt=vis_mesh(t_img,smpl_mesh_coord)
            mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
            img=np.ones((img_shape[0],img_shape[1],3))*255
            rendered_img=h36m_render_mesh(img,mesh_cam[0].cpu().numpy(),face,cam_param)
            ts=torch.FloatTensor(rendered_img)/255.
            ts=ts[None,:,:,:].cuda()
            ts=ts.transpose(1,3).transpose(2,3)
            tt=ced(ts)
            current_dir='s_{}_act_{}_subact_{}_ca_{}_{}.jpg'.format(subject,action_idx,subaction_idx,cam_idx,frame_idx)
            if threshold<=256:
                cv2.imwrite(data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx)+'/'+'SMPL_edge_'+current_dir,tt[0][0][0].cpu().numpy()*255)
            elif threshold<=4096 and threshold>256:
                cv2.imwrite(data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx)+'/'+'SMPL_edge_'+current_dir,tt[1][0][0].cpu().numpy()*255)
            elif threshold<=65536 and threshold>4096:
                cv2.imwrite(data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx)+'/'+'SMPL_edge_'+current_dir,tt[2][0][0].cpu().numpy()*255)
            elif threshold>65536:
                cv2.imwrite(data_path+'/SMPL_edge/'+'s_{}_act_{}_subact_{}_ca_{}'.format(subject,action_idx,subaction_idx,cam_idx)+'/'+'SMPL_edge_'+current_dir,tt[3][0][0].cpu().numpy()*255)
        except:
            pass
        

if __name__ =="__main__":
    make_h36m_smpl_Edge(data_path)