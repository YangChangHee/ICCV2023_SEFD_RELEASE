from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle
import kornia as K
from torch import Tensor
from kornia.color import bgr_to_rgb,rgb_to_grayscale
from kornia.filters import canny, gaussian_blur2d, motion_blur
from kornia.morphology import dilation
import torch
import torch.nn as nn
import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
class canny_edge(nn.Module):
    def __init__(self,apply_color_jitter: bool =False)-> None:
        super().__init__()
        self._apply_color_jitter=apply_color_jitter
        self.kernel1=torch.ones((5,5)).cuda()
        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)
        self.chshuffle=RandomChannelShuffle(p=0.75)

    @torch.no_grad() # No_grad
    def forward(self, x:Tensor) -> Tensor:
        x_out5=self.chshuffle(x)
        x=bgr_to_rgb(x)
        x=rgb_to_grayscale(x)
        x_out=canny(x)
        x_out2=dilation(x_out[0],kernel=self.kernel1)
        if self._apply_color_jitter:
            print(self._apply_color_jitter)
            x_out5 = self.jitter(x_out5)
        
        return x_out2


# putting your path ex) "/home/user/SEFD/dataset"
base_dir='putting your path/AGORA/SMPL_edge'
target_dir='putting your path/AGORA/SMPL_edge_55/'

if __name__ =="__main__":
    ced=canny_edge()
    list_dir=os.listdir(base_dir)
    for smpl_dir in tqdm(list_dir):
        smpl_path=osp.join(base_dir,smpl_dir)
        smpl=cv2.imread(smpl_path)
        this_list=smpl_dir.split(".png")
        smpl=smpl/255.
        smpl=torch.FloatTensor(smpl)
        smpl=smpl.unsqueeze(0)
        smpl=smpl.transpose(1,3).transpose(2,3)
        smpl=smpl.cuda()
        dilation_edge=ced(smpl)
        result=dilation_edge.cpu().numpy()
        result=np.minimum(1,result)
        cv2.imwrite(target_dir+this_list[0]+this_list[1],result[0][0]*255)