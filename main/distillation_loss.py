import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import adaptive_max_pool2d, normalize, cosine_similarity
from config import cfg

import torch.nn.functional as F

class ATLoss(nn.Module):
    """
    "Paying More Attention to Attention: Improving the Performance of
     Convolutional Neural Networks via Attention Transfer"
    Referred to https://github.com/szagoruyko/attention-transfer/blob/master/utils.py
    Discrepancy between Eq. (2) in the paper and the author's implementation
    https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L18-L23
    as partly pointed out at https://github.com/szagoruyko/attention-transfer/issues/34
    To follow the equations in the paper, use mode='paper' in place of 'code'
    """
    def __init__(self):
        super().__init__()
        self.mode = 'code'
        if self.mode not in ('code', 'paper'):
            raise ValueError('mode `{}` is not expected'.format(self.mode))

    @staticmethod
    def attention_transfer_paper(feature_map):
        return normalize(feature_map.pow(2).sum(1).flatten(1))

    def compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer_paper(student_feature_map)
        at_teacher = self.attention_transfer_paper(teacher_feature_map)
        return torch.norm(at_student - at_teacher, dim=1).sum()

    @staticmethod
    def attention_transfer(feature_map):
        return normalize(feature_map.pow(2).mean(1).flatten(1))

    def compute_at_loss(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer(student_feature_map)
        at_teacher = self.attention_transfer(teacher_feature_map)
        return (at_student - at_teacher).pow(2).mean()

    def forward(self, student_feature, teacher_feature):
        at_loss = 0
        batch_size = cfg.train_batch_size

        # list feature map student <=> teacher list

        factor = 1.0
        for i,j in zip(student_feature,teacher_feature):
            if self.mode == 'paper':
                at_loss += factor * self.compute_at_loss_paper(i, j)
            else:
                at_loss += factor * self.compute_at_loss(i, j)
            if batch_size is None:
                batch_size = len(i)

        return at_loss / batch_size * 100 if self.mode == 'paper' else at_loss * 100



class FSPLoss(nn.Module):
    """
    "A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning"
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_fsp_matrix(first_feature_map, second_feature_map):
        first_h, first_w = first_feature_map.shape[2:4]
        second_h, second_w = second_feature_map.shape[2:4]
        target_h, target_w = min(first_h, second_h), min(first_w, second_w)
        if first_h > target_h or first_w > target_w:
            first_feature_map = adaptive_max_pool2d(first_feature_map, (target_h, target_w))

        if second_h > target_h or second_w > target_w:
            second_feature_map = adaptive_max_pool2d(second_feature_map, (target_h, target_w))

        first_feature_map = first_feature_map.flatten(2)
        second_feature_map = second_feature_map.flatten(2)
        hw = first_feature_map.shape[2]
        return torch.matmul(first_feature_map, second_feature_map.transpose(1, 2)) / hw

    def forward(self, student_feature_map, teacher_feature_map):
        fsp_loss = 0
        batch_size = cfg.train_batch_size
        list_fps_pairs=[[0,1],[1,2],[2,3]]
        for fps_pairs in list_fps_pairs:
            student_fsp_matrices = self.compute_fsp_matrix(student_feature_map[fps_pairs[0]], student_feature_map[fps_pairs[1]])
            teacher_fsp_matrices = self.compute_fsp_matrix(teacher_feature_map[fps_pairs[0]], teacher_feature_map[fps_pairs[1]])
            factor = 1
            fsp_loss += factor * (student_fsp_matrices - teacher_fsp_matrices).norm(dim=1).sum()
        return fsp_loss / batch_size