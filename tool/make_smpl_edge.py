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

import time
from tqdm import tqdm

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

smpl = SMPL()
face = smpl.face
joint_regressor = smpl.joint_regressor
vertex_num = smpl.vertex_num
joint_num = smpl.joint_num
joints_name = smpl.joints_name
skeleton = smpl.skeleton
root_joint_idx = smpl.root_joint_idx
face_kps_vertex = smpl.face_kps_vertex


# Putting Path Data Directory
data_path = 'putting your path'

muco_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
muco_root_joint_idx = muco_joints_name.index('Pelvis')

def make_mpii_smpl_Edge(data_path):
    data_path=osp.join(data_path,"MPII")
    ced=canny_edge()
    if not os.path.exists(data_path+'/SMPL_edge'):
        os.makedirs(data_path+'/SMPL_edge')
        print("make of directory : "+data_path+'/SMPL_edge')
    if not os.path.exists(data_path+'/SMPL'):
        os.makedirs(data_path+'/SMPL')
        print("make of directory : "+data_path+'/SMPL')

    with open(osp.join(data_path,"annotations", 'MPII_train_SMPL_NeuralAnnot.json')) as f:
        smpl_params = json.load(f)

    db = COCO(osp.join(data_path,"annotations", 'train.json'))

    for iid in db.imgs.keys():

        aids = db.getAnnIds([iid])
        image_file_name=db.imgs[iid]['file_name']
        h1,w2=0,0
        list_mesh=[]
        list_rendered_edge=[]
        list_camera_parma=[]
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
                rendered_img=render_mesh(img,list_mesh[-1],face,person_smpl['cam_param'])
                list_name=image_file_name.split('/')
                cv2.imwrite(data_path+'/SMPL/'+'SMPL_'+list_name[1][:-4]+"_"+str(aid)+".jpg",rendered_img)
                ts=torch.FloatTensor(rendered_img)/255.
                ts=ts[None,:,:,:].cuda()
                ts=ts.transpose(1,3).transpose(2,3)
                tt=ced(ts)

                if threshold<=256:
                    cv2.imwrite(data_path+'/SMPL_edge/'+'SMPL_edge_'+list_name[1][:-4]+"_"+str(aid)+".jpg",tt[0][0][0].cpu().numpy()*255)
                elif threshold<=4096 and threshold>256:
                    cv2.imwrite(data_path+'/SMPL_edge/'+'SMPL_edge_'+list_name[1][:-4]+"_"+str(aid)+".jpg",tt[1][0][0].cpu().numpy()*255)
                elif threshold<=65536 and threshold>4096:
                    cv2.imwrite(data_path+'/SMPL_edge/'+'SMPL_edge_'+list_name[1][:-4]+"_"+str(aid)+".jpg",tt[2][0][0].cpu().numpy()*255)
                elif threshold>65536:
                    cv2.imwrite(data_path+'/SMPL_edge/'+'SMPL_edge_'+list_name[1][:-4]+"_"+str(aid)+".jpg",tt[3][0][0].cpu().numpy()*255)

def multi_mpii_smpl_Edge(data_path):
    data_path=osp.join(data_path,"MPII")
    ced=canny_edge()
    if not os.path.exists(data_path+'/SMPL_occ_edge'):
        os.makedirs(data_path+'/SMPL_occ_edge')
        print("make of directory : "+data_path+'/SMPL_occ_edge')

    if not os.path.exists(data_path+'/SMPL_multi_edge'):
        os.makedirs(data_path+'/SMPL_multi_edge')
        print("make of directory : "+data_path+'/SMPL_multi_edge')

    if not os.path.exists(data_path+'/SMPL_multi'):
        os.makedirs(data_path+'/SMPL_multi')
        print("make of directory : "+data_path+'/SMPL_multi')
    
    if not os.path.exists(data_path+'/SMPL_test'):
        os.makedirs(data_path+'/SMPL_test')
        print("make of directory : "+data_path+'/SMPL_test')

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
                cv2.imwrite(data_path+'/SMPL_test/'+'SMPL_test_'+list_name[1][:-4]+'{}.jpg'.format(n),tt[1][0][0].cpu().numpy()*255)
                list_rendered_edge.append(tt)

            img2=np.ones((h1,w2,3))*255
            result=mscoco_multi_render_mesh(img2,list_mesh,face,list_camera_parma)
            cv2.imwrite(data_path+'/SMPL_multi/'+'SMPL_multi_'+list_name[1],result)
            ts1=torch.FloatTensor(result)/255.
            ts1=ts1[None,:,:,:].cuda()
            ts1=ts1.transpose(1,3).transpose(2,3)
            tt1=ced(ts1)
            test=np.zeros((h1,w2))
            for j in list_rendered_edge:
                test+=j[1][0][0].cpu().numpy()
            result1=np.maximum(test-tt1[1][0][0].cpu().numpy(),0)
            result1=np.minimum(result1,1)

            cv2.imwrite(data_path+'/SMPL_multi_edge/'+'SMPL_multi_edge_'+list_name[1],tt1[1][0][0].cpu().numpy()*255)
            cv2.imwrite(data_path+'/SMPL_occ_edge/'+'SMPL_occ_edge_'+list_name[1],result1*255)
        else:
            pass
        #else:
        #    print("True")

def make_mscoco_smpl_Edge(data_path):
    data_path=osp.join(data_path,"MSCOCO")
    ced=canny_edge()
    if not os.path.exists(data_path+'/images/SMPL_edge'):
        os.makedirs(data_path+'/images/SMPL_edge')
        print("make of directory : "+data_path+'/images/SMPL_edge')
    if not os.path.exists(data_path+'/images/SMPL'):
        os.makedirs(data_path+'/images/SMPL')
        print("make of directory : "+data_path+'/images/SMPL')

    with open(osp.join(data_path,"annotations", 'MSCOCO_train_SMPL_NeuralAnnot.json')) as f:
        smpl_params = json.load(f)
    db = COCO(osp.join(data_path,"annotations", 'person_keypoints_' + 'train' + '2017.json'))

    for iid in tqdm(db.imgs.keys()):

        aids = db.getAnnIds([iid])
        image_file_name=db.imgs[iid]['file_name']
        h1,w2=0,0
        list_mesh=[]
        list_rendered_edge=[]
        list_camera_parma=[]
        for aid in aids:
            ann=db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            h1,w2=img['height'], img['width']
            threshold=ann['bbox'][2]*ann['bbox'][3]
            img=np.ones((h1,w2,3))*255
            if not os.path.exists(data_path+'/images/SMPL_edge'+'/'+'SMPL_edge_'+image_file_name[:-4]+"_"+str(aid)+".jpg"):
                if str(aid) in smpl_params:
                    person_smpl=smpl_params[str(aid)]
                    smpl_mesh_coord, _ =get_smpl_coord(person_smpl['smpl_param'],benchmark='mscoco')
                    list_camera_parma.append(person_smpl['cam_param'])
                    mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
                    list_mesh.append(mesh_cam[0].cpu().numpy())
                    rendered_img=render_mesh(img,list_mesh[-1],face,person_smpl['cam_param'])
                    
                    cv2.imwrite(data_path+'/images/SMPL'+'/'+'SMPL'+image_file_name[:-4]+"_"+str(aid)+".jpg",rendered_img)
                    ts=torch.FloatTensor(rendered_img)/255.
                    
                    ts=ts[None,:,:,:].cuda()
                    ts=ts.transpose(1,3).transpose(2,3)
                    tt=ced(ts)
                    if threshold<=256:
                        cv2.imwrite(data_path+'/images/SMPL_edge'+'/'+'SMPL_edge_'+image_file_name[:-4]+"_"+str(aid)+".jpg",tt[0][0][0].cpu().numpy()*255)
                    elif threshold<=4096 and threshold>256:
                        cv2.imwrite(data_path+'/images/SMPL_edge'+'/'+'SMPL_edge_'+image_file_name[:-4]+"_"+str(aid)+".jpg",tt[1][0][0].cpu().numpy()*255)
                    elif threshold<=65536 and threshold>4096:
                        cv2.imwrite(data_path+'/images/SMPL_edge'+'/'+'SMPL_edge_'+image_file_name[:-4]+"_"+str(aid)+".jpg",tt[2][0][0].cpu().numpy()*255)
                    elif threshold>65536:
                        cv2.imwrite(data_path+'/images/SMPL_edge'+'/'+'SMPL_edge_'+image_file_name[:-4]+"_"+str(aid)+".jpg",tt[3][0][0].cpu().numpy()*255)
                    

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
                if not os.path.exists(data_path+'/3dpw/SMPL_edge/'+currnet_dir):
                    os.makedirs(data_path+'/3dpw/SMPL_edge/'+currnet_dir)
                    print("make of directory : " +data_path+'/3dpw/SMPL_edge/'+currnet_dir)
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
                        result=np.maximum(test-tt1[0][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'occlusion_masking_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'SMPL_edge_'+current_path,tt1[0][0][0].cpu().numpy()*255)
                    elif threshold<=4096 and threshold>256:
                        for j in list_rendered_edge:
                            test+=j[1][0][0].cpu().numpy()
                        result=np.maximum(test-tt1[1][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'occlusion_masking_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'SMPL_edge_'+current_path,tt1[1][0][0].cpu().numpy()*255)
                    elif threshold<=65536 and threshold>4096:
                        for j in list_rendered_edge:
                            test+=j[2][0][0].cpu().numpy()
                        result=np.maximum(test-tt1[2][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'occlusion_masking_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'SMPL_edge_'+current_path,tt1[2][0][0].cpu().numpy()*255)
                    elif threshold>65536:
                        for j in list_rendered_edge:
                            test+=j[3][0][0].cpu().numpy()
                        result=np.maximum(test-tt1[3][0][0].cpu().numpy(),0)
                        result=np.minimum(result,1)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'occlusion_masking_'+current_path,result*255)
                        cv2.imwrite(data_path+'/3dpw/SMPL_edge/'+currnet_dir+'/'+'SMPL_edge_'+current_path,tt1[3][0][0].cpu().numpy()*255)
                    
                    
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

def make_3dpw_smpl(data_path):
    ced=canny_edge()
    if not os.path.exists(data_path+'/3dpw/SMPL/'):
        os.makedirs(data_path+'/3dpw/SMPL/')
        print("make of directory : "+data_path+'/3dpw/SMPL/')
    db = COCO(osp.join(data_path,'3dpw','data', '3DPW_latest_' + 'test' + '.json')) # validation, test
    person_num=0
    current_path=''
    currnet_dir=''
    list_mesh=[]
    list_rendered_edge=[]
    check_dir=['downtown_cafe_00']
#    check_dir=['outdoors_fencing_01','downtown_warmWelcome_00','downtown_walking_00','flat_packBags_00','downtown_windowShopping_00']
    before_w,before_h=0,0
    for i in range(len(db.anns)):
        #if db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] == "downtown_runForBus_01" or db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] == "downtown_walkBridge_01" or db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] == "downtown_walkUphill_00" or db.loadImgs(db.anns[i]['image_id'])[0]['sequence'] ==  "flat_guitar_01" :
        if False: # test
            pass
        else:
            if currnet_dir != db.loadImgs(db.anns[i]['image_id'])[0]['sequence']:
                currnet_dir=db.loadImgs(db.anns[i]['image_id'])[0]['sequence']
                if not os.path.exists(data_path+'/3dpw/SMPL/'+currnet_dir):
                    os.makedirs(data_path+'/3dpw/SMPL/'+currnet_dir)
                    print("make of directory : " +data_path+'/3dpw/SMPL/'+currnet_dir)
            if currnet_dir not in check_dir:
                continue
            mesh,w,h=db.anns[i]['smpl_param'],db.loadImgs(db.anns[i]['image_id'])[0]['width'], db.loadImgs(db.anns[i]['image_id'])[0]['height']
            smpl_mesh_coord, _=get_smpl_coord(mesh,benchmark='normal')
            #save_obj(smpl_mesh_coord,face,'test.obj_{}'.format(i))
            person_num+=1
            if current_path!=db.loadImgs(db.anns[i]['image_id'])[0]['file_name']:
                current_path=db.loadImgs(db.anns[i]['image_id'])[0]['file_name']
                person_num=0
            mesh_cam=smpl_mesh_coord
            w,h=db.loadImgs(db.anns[i]['image_id'])[0]['width'], db.loadImgs(db.anns[i]['image_id'])[0]['height']
            img=np.ones((h,w,3))*255
            rendered_img2 = render_mesh(img, mesh_cam, face, db.loadImgs(db.anns[i]['image_id'])[0]['cam_param'])

            cv2.imwrite(data_path+'/3dpw/SMPL/'+currnet_dir+'/'+'SMPL_'+current_path[:-4]+"_"+str(person_num)+".jpg",rendered_img2)

                    


def inverse_make_muco_smpl_Edge(data_path):
    data_path=osp.join(data_path,"MuCo")
    ced=canny_edge()
    if not os.path.exists(data_path+'/SMPL_edge'):
        os.makedirs(data_path+'/SMPL_edge')
        print("make of directory : "+data_path+'/SMPL_edge')
    
    smpl_param_path = osp.join(data_path, 'data', 'smpl_param.json')
    annot_path = osp.join(data_path, 'data', 'MuCo-3DHP.json')
    db = COCO(annot_path)
    with open(smpl_param_path) as f:
        smpl_params = json.load(f)
    
    #db=reversed(db)
    
    for iid in db.imgs.keys():
        img = db.imgs[len(db.imgs.keys())-1-iid]
        img_id = img["id"]
        img_width, img_height = img['width'], img['height']
        imgname = img['file_name']
        list_split=imgname.split('/')
        focal = img["f"]
        princpt = img["c"]
        cam_param = {'focal': focal, 'princpt': princpt}
        #print(cam_param)
        #if list_split[0]=='unaugmented_set':
        if int(list_split[1])>=166 and int(list_split[1]) <=167:
            if not os.path.exists(data_path+'/SMPL_edge/'+list_split[0]+'/'+list_split[1]):
                os.makedirs(data_path+'/SMPL_edge/'+list_split[0]+'/'+list_split[1])
                print("make of directory : "+data_path+'/SMPL_edge/'+list_split[0]+'/'+list_split[1])
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)
            current_path=list_split[2]
            list_mesh=[]
            list_rendered_edge=[]
            for pid in ann_ids:
                try:
                    smpl_param = smpl_params[str(pid)]
                    smpl_mesh_coord, _=get_smpl_coord(smpl_param,benchmark='mscoco')
                    img=np.ones((img_height,img_width,3))*255
                    mesh_cam=torch.Tensor(smpl_mesh_coord[None,:,:]).cuda()
                    list_mesh.append(mesh_cam[0].cpu().numpy())
                    rendered_img=render_mesh(img,list_mesh[-1],face,cam_param)
                    
                    #cv2.imwrite("putting your path/Project/3DCrowdNet_RELEASE/tool/test.jpg",rendered_img)
                    ts=torch.FloatTensor(rendered_img)/255.
                    ts=ts[None,:,:,:].cuda()
                    ts=ts.transpose(1,3).transpose(2,3)
                    tt=ced(ts)
                    list_rendered_edge.append(tt)
                    cv2.imwrite(data_path+'/SMPL_edge/'+list_split[0]+'/'+list_split[1]+'/'+'SMPL_edge_{}_'.format(pid)+current_path,tt[2][0][0].cpu().numpy()*255)
                except:
                    pass
            img1=np.ones((img_height,img_width,3))*255
            rendered_img2 = multi_render_mesh(img1, list_mesh, face, cam_param)
            ts1=torch.FloatTensor(rendered_img2)/255.
            ts1=ts1[None,:,:,:].cuda()
            ts1=ts1.transpose(1,3).transpose(2,3)
            tt1=ced(ts1)
            cv2.imwrite(data_path+'/SMPL_edge/'+list_split[0]+'/'+list_split[1]+'/'+'full_'+current_path,tt1[2][0][0].cpu().numpy()*255)
        else:
            pass
import pickle
import matplotlib.pyplot as plt
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
    #make_3dpw_smpl_Edge(data_path)
    #make_mscoco_smpl_Edge(data_path)
    #make_mpii_smpl_Edge(data_path)
    #multi_mpii_smpl_Edge(data_path)
    #inverse_make_muco_smpl_Edge(data_path)
    #AGORA_smpl_edge(data_path)
    #multi_mpii_smpl_Edge(data_path)
    make_3dpw_smpl(data_path)