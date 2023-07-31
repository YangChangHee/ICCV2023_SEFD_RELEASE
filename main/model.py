import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import Pose2Feat, PositionNet, RotationNet, Vposer
from nets.UNet import UNet
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math
from utils.transforms import rot6d_to_axis_angle

import cv2

from mmcv.cnn import constant_init, kaiming_init
from utils.transforms import rot6d_to_axis_angle
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle
import kornia as K
from torch import Tensor
from kornia.color import bgr_to_rgb,rgb_to_grayscale
from kornia.filters import canny, gaussian_blur2d, motion_blur
from kornia.morphology import dilation



class canny_edge(nn.Module):
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._apply_color_jitter = apply_color_jitter
        self.kernel=torch.ones(cfg.dilation[0]).to(device=device)
        self.kernel1=torch.ones(cfg.dilation[1]).to(device=device)
        self.kernel2=torch.ones(cfg.dilation[2]).to(device=device)
        self.kernel3=torch.ones(cfg.dilation[3]).to(device=device)

        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)
        self.chshuffle=RandomChannelShuffle(p=0.75)

    @torch.no_grad() # No_grad
    def forward(self, x:Tensor) -> Tensor:
        x_out5=self.chshuffle(x)
        x=bgr_to_rgb(x)
        x=rgb_to_grayscale(x)
        x_out=canny(x)
#        x_out_t=x_out[0].cuda ()
        x_out1=dilation(x_out[0],kernel=self.kernel)
        x_out2=dilation(x_out[0],kernel=self.kernel1)
        x_out3=dilation(x_out[0],kernel=self.kernel2)
        x_out4=dilation(x_out[0],kernel=self.kernel3)
        if self._apply_color_jitter:
            print(self._apply_color_jitter)
            x_out5 = self.jitter(x_out5)
        
        return x_out[0], x_out1,x_out2,x_out3,x_out4, x_out5
class smpl_model(nn.Module):
    def __init__(self, backbone, pose2feat):
        super(smpl_model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float();
        yy = yy[None, None, :, :].cuda().float();

        x = joint_coord_img[:, :, 0, None, None];
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap


    def forward(self, inputs, mode):
        with torch.no_grad():
            concat_input=torch.cat([inputs['img'],inputs['SMPL_edge']],dim=1)
            feature1 = self.backbone(concat_input)
            joint_coord_img = inputs['joints']
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
            feature2 = self.pose2feat(feature1, joint_heatmap)
            f1,f2,f3,f4 = self.backbone(feature2, skip_early=True)  # 2048 x 8 x 8
        if mode == 'train':
            return f1, f2, f3,f4


class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net,smpl_overlap_model,edge_module, vposer):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer
        self.ced=canny_edge()
        
        

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer.cuda()
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    def get_camera_trans(self, cam_param, meta_info, is_render):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_img_shape[0]*cfg.input_img_shape[1]))]).cuda().view(-1)
        if is_render:
            bbox = meta_info['bbox']
            k_value = k_value * math.sqrt(cfg.input_img_shape[0]*cfg.input_img_shape[1]) / (bbox[:, 2]*bbox[:, 3]).sqrt()
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float();
        yy = yy[None, None, :, :].cuda().float();

        x = joint_coord_img[:, :, 0, None, None];
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
        root_joint_idx = self.human_model.root_joint_idx

        # project 3D coordinates to 2D space
        x = joint_cam[:,:,0] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]
        y = joint_cam[:,:,1] / (joint_cam[:,:,2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        mesh_cam_render = mesh_cam.clone()
        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def forward(self, inputs, targets, meta_info, mode):
        _,_,x_out3,_,_,_=self.ced(inputs['img'])
        concat_input1=torch.cat([inputs['img'],x_out3],dim=1)

        early_img_feat = self.backbone(concat_input1)
            
    
        joint_coord_img = inputs['joints']
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
        _,_,_,pose_guided_img_feat  = self.backbone(pose_img_feat, skip_early=True)

        joint_img, joint_score = self.position_net(pose_guided_img_feat) 

        root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach())
        root_pose = rot6d_to_axis_angle(root_pose_6d)
        pose_param = self.vposer(z)
        cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
        joint_proj, _, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)

        out = {'cam_param': cam_param}
        out['input_joints'] = joint_coord_img
        out['joint_img'] = joint_img * 8
        out['joint_proj'] = joint_proj
        out['joint_score'] = joint_score
        out['smpl_mesh_cam'] = mesh_cam
        out['smpl_pose'] = pose_param
        out['smpl_shape'] = shape_param

        out['mesh_cam_render'] = mesh_cam_render
        if cfg.demo==True:
            out['canny_edge']= x_out3

        if 'smpl_mesh_cam' in targets:
            out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
        if 'bb2img_trans' in meta_info:
            out['bb2img_trans'] = meta_info['bb2img_trans']
        if 'img2bb_trans' in meta_info:
            out['img2bb_trans'] = meta_info['img2bb_trans']
        if 'bbox' in meta_info:
            out['bbox'] = meta_info['bbox']
        if 'tight_bbox' in meta_info:
            out['tight_bbox'] = meta_info['tight_bbox']
        if 'aid' in meta_info:
            out['aid'] = meta_info['aid']

        return out

def smpl_get_model(joint_num):
    backbone = ResNetBackbone(cfg.resnet_type,is_smpl=True)
    pose2feat=Pose2Feat(joint_num)
    model = smpl_model(backbone,pose2feat)
    return model

def get_model(vertex_num, joint_num,smpl_overlap_model, mode):

    backbone = ResNetBackbone(cfg.resnet_type,is_smpl=False)
    
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()
    edge_module=None

    
    backbone.init_weights()

    model = Model(backbone, pose2feat, position_net, rotation_net,smpl_overlap_model,edge_module, vposer)

    return model

