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
from pidinet_config import config_model, config_model_converted
from Pidinet import PiDiNet
from RCF import RCF
import cv2
from prior import MaxMixturePrior
import sys
sys.path.append("/home/qazw5741/SEFD/common/")
from nets.cls_hrnet import HighResolutionNet,get_cls_net
from nets.hrnet_config import cfg as cfg_t
from nets.hrnet_config import update_config

from mmcv.cnn import constant_init, kaiming_init
from utils.transforms import rot6d_to_axis_angle
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle
import kornia as K
from torch import Tensor
from kornia.color import bgr_to_rgb,rgb_to_grayscale
from kornia.filters import canny, gaussian_blur2d, motion_blur
from kornia.morphology import dilation
from distillation_loss import ATLoss, FSPLoss
import sys

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True



class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=2):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class Gaussian_Blur(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gaus_blur=gaussian_blur2d
        self.motion_blur=motion_blur

    @torch.no_grad() # No_grad
    def forward(self,x):
        x=self.motion_blur(x,  7, 90., 1)
        #print(x.shape)
        return x


class HED_Edge(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-hed/network-' + 'bsds500' + '.pytorch', file_name='hed-' + 'bsds500').items() })
    # end

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([ tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv ], 1))



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


        
def pidinet_converted():
    if cfg.dilstillation_edge=="PiDiNet" or cfg.pidinet_edge==True:
        pidinet_config="carv4"
    pdcs = config_model_converted(pidinet_config)
    dil = 24 if True else None
    return PiDiNet(60, pdcs, dil=dil, sa=True, convert=True)

def pidinet():
    if cfg.dilstillation_edge=="PiDiNet" or cfg.pidinet_edge==True:
        pidinet_config="carv4"
    pdcs = config_model(pidinet_config)
    dil = 24 if True else None
    return PiDiNet(60, pdcs, dil=dil, sa=True)




class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net,smpl_overlap_model,edge_module, vposer):
        super(Model, self).__init__()
        self.backbone = backbone
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer
        self.ced=canny_edge()
        if cfg.dilstillation_edge=="Canny" or cfg.canny_edge ==True:
            self.ced=canny_edge()
        elif cfg.dilstillation_edge=="Hed" or cfg.HED_edge==True:
            self.hed_edge=HED_Edge()
            print("Load hed_edge model complete")
        elif cfg.dilstillation_edge=="PiDiNet" or cfg.pidinet_edge==True:
            self.pidinet = pidinet()
            self.pidinet = torch.nn.DataParallel(self.pidinet).cuda()
            self.pidinet.load_state_dict(torch.load("putting your path/SEFD/table5_pidinet.pth")['state_dict'])
            print("Load pidinet_edge model complete")
        elif cfg.dilstillation_edge=="RCF" or cfg.srf_edge==True:
            self.RCF_edge=RCF()
            self.RCF_edge.load_state_dict(torch.load(cfg.RCF_path)['state_dict'])
            print("Load RCF_edge model complete")
        elif cfg.dilstillation_edge=="edge_module":
            #self.ced=canny_edge()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.edge_module=UNet(n_channels=3, n_classes=2, bilinear=False,model_version="2").to(device=device)
            self.edge_module.load_state_dict(torch.load(cfg.edge_module_path, map_location=device))


        self.gau_blur=Gaussian_Blur()
        if cfg.edge_module==True:
            self.edge_module1 = edge_module
        if cfg.distillation_module==True:
            self.smpl_overlap_backbone=smpl_overlap_model
        
        

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
        self.l1_loss=nn.L1Loss()
        self.kl_loss=nn.KLDivLoss(reduction="batchmean")
        if cfg.distillation_loss=="ATLoss":
            self.atloss=ATLoss()
        elif cfg.distillation_loss=="GC_loss":
            self.GCBLOCK1=ContextBlock2d(256,256)
            self.GCBLOCK2=ContextBlock2d(512,512)
            self.GCBLOCK3=ContextBlock2d(1024,1024)
            self.GCBLOCK4=ContextBlock2d(2048,2048)
        elif cfg.distillation_loss=="fsp_loss":
            self.fsploss=FSPLoss()
            print("Load_FSP_Success")
        elif cfg.distillation_loss=="KD_loss":
            self.KD_loss = nn.KLDivLoss(reduction="batchmean")
        #self.vgg16_loss=VGGPerceptualLoss().cuda()

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
        if cfg.grid_plus==True:
            x_out0,x_out1,x_out2,x_out3,x_out4, x_out5=self.ced(inputs['img'])
            concat_input=torch.cat([inputs['img'],x_out2],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.canny_edge==True:
            x_out0,x_out1,x_out2,x_out3,x_out4, x_out5=self.ced(inputs['img'])
            concat_input=torch.cat([inputs['img'],x_out2],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.human_parsing==True:
            concat_input=torch.cat([inputs['img'],inputs['parsing']],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.srf_edge==True:
            rcf_edge=self.RCF_edge(inputs['test_img'])
            concat_input=torch.cat([inputs['img'],rcf_edge[-1]],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.pidinet_edge==True:
            test=self.pidinet(inputs['img'])#inputs['pidinet_img'])
            pidinet_edge=test[-1]
            concat_input1=torch.cat([inputs['img'],pidinet_edge],dim=1)
            early_img_feat = self.backbone(concat_input1)
        elif cfg.EPS_plus==True:
            if cfg.EPS_grid_plus==True:
                concat_input=torch.cat([inputs['img'],inputs['EPS'],inputs['grid']],dim=1)
            elif cfg.EPS_edge_plus==True:
                x_out0,x_out1,x_out2,x_out3,x_out4, x_out5=self.ced(inputs['img'])
                concat_input=torch.cat([x_out5,inputs['EPS'],x_out3],dim=1)
            else:
                concat_input=torch.cat([inputs['img'],inputs['EPS']],dim=1)

            early_img_feat = self.backbone(concat_input)  #pose_guided_img_feat
        elif cfg.HED_edge==True:
            hed_edge=self.hed_edge(inputs['img'])
            concat_input=torch.cat([inputs['img'],hed_edge],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.SMPL_edge==True and cfg.distillation_module == False and cfg.nothing==False:
            concat_input=torch.cat([inputs['img'],inputs['smpl_edge']],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.edge_module==True:
            with torch.no_grad():
                edge,seg=self.edge_module1(inputs['img'])
                estimation_smpl_edge=torch.sigmoid(edge)
                estimation_smpl_seg=torch.sigmoid(seg)
            concat_input=torch.cat([inputs['img'],estimation_smpl_edge],dim=1)
            early_img_feat = self.backbone(concat_input)
        elif cfg.distillation_module:
            with torch.no_grad():
                if mode =="train":
                    feature1, feature2, feature3,feature4=self.smpl_overlap_backbone(inputs,"train")
            if cfg.dilstillation_edge=="edge_module":
                edge,seg=self.edge_module(inputs['img'])
                estimation_smpl_edge=torch.sigmoid(edge)
                estimation_smpl_seg=torch.sigmoid(seg)
                concat_input1=torch.cat([inputs['img'],estimation_smpl_edge],dim=1)
            if cfg.dilstillation_edge=="Canny":
                _,_,x_out3,_,_,_=self.ced(inputs['img'])
                concat_input1=torch.cat([inputs['img'],x_out3],dim=1)
            elif cfg.dilstillation_edge=="Hed":
                hed_edge=self.hed_edge(inputs['img'])
                concat_input1=torch.cat([inputs['img'],hed_edge],dim=1)
            elif cfg.dilstillation_edge=="PiDiNet":
                test=self.pidinet(inputs['img'])
                pidinet_edge=test[-1]
                concat_input1=torch.cat([inputs['img'],pidinet_edge],dim=1)
            elif cfg.dilstillation_edge=="RCF":
                rcf_edge=self.RCF_edge(inputs['test_img'])
                concat_input1=torch.cat([inputs['img'],rcf_edge[-1]],dim=1)

            early_img_feat = self.backbone(concat_input1)
        else:
            early_img_feat = self.backbone(inputs['img'])
            
        if cfg.distillation_module==False:
            joint_coord_img = inputs['joints']
            with torch.no_grad():
                joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
                joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
            
            pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
            pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)  

            joint_img, joint_score = self.position_net(pose_guided_img_feat)  

            # estimate model parameters
            root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach())
            # change root pose 6d + latent code -> axis angles
            root_pose = rot6d_to_axis_angle(root_pose_6d)
            pose_param = self.vposer(z)
            cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))
            pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
            pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
            joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)
        elif cfg.distillation_module==True:
            #print(inputs.keys())
            joint_coord_img = inputs['joints']
            with torch.no_grad():
                joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())
                joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]
            pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)
            b_feature_1,b_feature_2,b_feature_3,pose_guided_img_feat  = self.backbone(pose_img_feat, skip_early=True)

            joint_img, joint_score = self.position_net(pose_guided_img_feat) 

            root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach())
            root_pose = rot6d_to_axis_angle(root_pose_6d)
            pose_param = self.vposer(z)
            cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))
            pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)
            pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)
            joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)

        if mode == 'train':
            # loss functions
            loss = {}
            # joint_img: 0~8, joint_proj: 0~64, target: 0~64
            loss['body_joint_img'] = (1/8) * self.coord_loss(joint_img*8, self.human_model.reduce_joint_set(targets['orig_joint_img']), self.human_model.reduce_joint_set(meta_info['orig_joint_trunc']), meta_info['is_3D'])
            loss['smpl_joint_img'] = (1/8) * self.coord_loss(joint_img*8, self.human_model.reduce_joint_set(targets['fit_joint_img']),
                                                    self.human_model.reduce_joint_set(meta_info['fit_joint_trunc']) * meta_info['is_valid_fit'][:, None, None])
            loss['smpl_pose'] = self.param_loss(pose_param, targets['pose_param'], meta_info['fit_param_valid'] * meta_info['is_valid_fit'][:, None])
            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = (1/8) * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2], meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:, None, None])
            if cfg.distillation_loss=="L1":
                loss['l1_loss_1'] = self.l1_loss(b_feature_1,feature1)
                loss['l1_loss_2'] = self.l1_loss(b_feature_2,feature2)
                loss['l1_loss_3'] = self.l1_loss(b_feature_3,feature3)
                loss['l1_loss_4'] = self.l1_loss(pose_guided_img_feat,feature4)
            elif cfg.distillation_loss=="ATLoss":
                atl=self.atloss([b_feature_1,b_feature_2,b_feature_3,pose_guided_img_feat],[feature1,feature2,feature3,feature4])
                loss['ATLoss']=atl
            elif cfg.distillation_loss=="log_softmax_l1":
                #b_feature_1=F.log_softmax(b_feature_1,dim=1)
                #feature1=F.log_softmax(feature1,dim=1)
                #b_feature_2=F.log_softmax(b_feature_2,dim=1)
                #feature2=F.log_softmax(feature2,dim=1)
                b_feature_3=F.log_softmax(b_feature_3,dim=1)
                feature3=F.log_softmax(feature3,dim=1)
                b_feature_4=F.log_softmax(pose_guided_img_feat,dim=1)
                feature4=F.log_softmax(feature4,dim=1)
                #loss['kd_feature_loss_1'] = self.l1_loss(b_feature_1,feature1)
                #loss['kd_feature_loss_2'] = self.l1_loss(b_feature_2,feature2)
                loss['logsoftmax_feature_3'] = self.l1_loss(b_feature_3,feature3)
                loss['logsoftmax_feature_4'] = self.l1_loss(b_feature_4,feature4) *2
            elif cfg.distillation_loss=="GC_loss":
                # loss가 안줄어들면 log_softmax(x, dim=1)이런식으로 바꿔야함
                new_feature1=self.GCBLOCK1(feature1)
                new_feature2=self.GCBLOCK2(feature2)
                new_feature3=self.GCBLOCK3(feature3)
                new_feature4=self.GCBLOCK4(feature4)
                new_b_feature_1=self.GCBLOCK1(b_feature_1)
                new_b_feature_2=self.GCBLOCK2(b_feature_2)
                new_b_feature_3=self.GCBLOCK3(b_feature_3)
                new_b_feature_4=self.GCBLOCK4(pose_guided_img_feat)
                loss['GC_feature_loss_1'] = self.l1_loss(new_b_feature_1,new_feature1)
                loss['GC_feature_loss_2'] = self.l1_loss(new_b_feature_2,new_feature2)
                loss['GC_feature_loss_3'] = self.l1_loss(new_b_feature_3,new_feature3)
                loss['GC_feature_loss_4'] = self.l1_loss(new_b_feature_4,new_feature4)
            elif cfg.distillation_loss=="fsp_loss":
                distil_loss=self.fsploss([b_feature_1,b_feature_2,b_feature_3,pose_guided_img_feat],[feature1,feature2,feature3,feature4])
                loss['fsp_loss']=distil_loss
            elif cfg.distillation_loss=="KD_loss":
                feature1=F.softmax(feature1,dim=1)
                feature2=F.softmax(feature2,dim=1)
                feature3=F.softmax(feature3,dim=1)
                feature4=F.softmax(feature4,dim=1)
                b_feature_1=F.log_softmax(b_feature_1, dim=1)
                b_feature_2=F.log_softmax(b_feature_2, dim=1)
                b_feature_3=F.log_softmax(b_feature_3, dim=1)
                pose_guided_img_feat=F.log_softmax(pose_guided_img_feat, dim=1)
                loss['KD_feature_loss_1']=self.KD_loss(b_feature_1,feature1)
                loss['KD_feature_loss_2']=self.KD_loss(b_feature_2,feature2)
                loss['KD_feature_loss_3']=self.KD_loss(b_feature_3,feature3)
                loss['KD_feature_loss_4']=self.KD_loss(pose_guided_img_feat,feature4)
            return loss

        else:
            # test output
            out = {'cam_param': cam_param}
            out['input_joints'] = joint_coord_img
            out['joint_img'] = joint_img * 8
            out['joint_proj'] = joint_proj
            out['joint_score'] = joint_score
            out['smpl_mesh_cam'] = mesh_cam
            out['smpl_pose'] = pose_param
            out['smpl_shape'] = shape_param
            out['joint_cam']=joint_cam #
            out['cam_trans']=cam_trans #
            #if cfg.feature_vis==True:
            #    out['feature'] = feature
            #else:
            #    out['feature']=None
            #시각화
            #out['edge']=hed_edge.clone().detach()

            out['mesh_cam_render'] = mesh_cam_render
            #if cfg.demo==True:
            #    out['canny_edge']= x_out3

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

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def smpl_get_model(joint_num):
    backbone = ResNetBackbone(cfg.resnet_type,is_smpl=True)
    pose2feat=Pose2Feat(joint_num)
    model = smpl_model(backbone,pose2feat)
    return model

def get_model(vertex_num, joint_num,smpl_overlap_model, mode):
    if cfg.Backbone_model=='resnet':
        backbone = ResNetBackbone(cfg.resnet_type,is_smpl=False)
        if cfg.nothing==True:
            backbone.init_weights()
    elif cfg.Backbone_model=='hrnet':
        import os.path as osp
        config_file = osp.join("/home/qazw5741/SEFD/common/nets/models/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        update_config(cfg_t, config_file)
        backbone = get_cls_net(cfg_t)
        
    
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()
    if cfg.edge_module:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        edge_module=UNet(n_channels=3, n_classes=2, bilinear=False,model_version="2").to(device=device)
        edge_module.load_state_dict(torch.load(cfg.edge_module_path, map_location=device))
        print("load_edge_module succeess!")
    else:
        edge_module=None
    if mode == 'train':
        if cfg.smpl_edge_test !=True:
            if cfg.distillation_pretrained==True or cfg.inference_testing==True:
                backbone.init_weights()
            elif cfg.SMPL_edge==True and cfg.SMPL_overlap==True and cfg.distillation_pretrained==False and cfg.teacher_learing==False:
                backbone.init_weights()
            elif cfg.first_input==True and cfg.inference_testing==True:
                backbone.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)
    if cfg.demo==True:
        backbone.init_weights()

    if cfg.edge_module:
        model = Model(backbone, pose2feat, position_net, rotation_net,smpl_overlap_model,edge_module, vposer)

    elif cfg.distillation_module:
        model = Model(backbone, pose2feat, position_net, rotation_net,smpl_overlap_model,edge_module, vposer)

    else:
        model = Model(backbone, pose2feat, position_net, rotation_net,None,None, vposer)
    return model

