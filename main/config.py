import os
import os.path as osp
import sys
import numpy as np
import datetime
import yaml
import shutil
import glob
from easydict import EasyDict as edict

class Config:
    ## dataset
    # MuCo, Human36M, MSCOCO, PW3D, FreiHAND
    trainset_3d = ['Human36M','MuCo']  # 'Human36M', 'MuCo'
    trainset_2d = ['MSCOCO','MPII','CrowdPose']  # 'MSCOCO', 'MPII', 'CrowdPose'
    #trainset_3d=['Human36M']
    #trainset_2d=[]
    testset = 'MuPoTs'  # 'MuPoTs' 'MSCOCO' Human36M, MSCOCO, 'PW3D'

    ## model setting
    resnet_type = 50  # 50, 101, 152
    
    ## input, output
    input_img_shape = (256, 256)  #(256, 192)
    output_hm_shape = (64, 64, 64)  #(64, 64, 48)
    bbox_3d_size = 2 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 0.3
    sigma = 2.5
    focal = (5000, 5000)  # virtual focal lengths
    princpt = (input_img_shape[1] / 2, input_img_shape[0] / 2)  # virtual principal point position

    ## training config
    lr_dec_epoch = [15] if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else [17,21]
    end_epoch = 20 #13 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 25
    lr = 1e-4
    lr_backbone = 1e-4
    lr_dec_factor = 10
    train_batch_size = 64
    use_gt_info = True

    ## testing config
    test_batch_size = 32
    crowd = False
    vis = False
    render = False
    origin= False

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    demo=True
    teacher_learing=False
    
    #patch
    canny_edge=False
    HED_edge=False
    pidinet_edge=False
    srf_edge=False
    luperson=False
    patch_size=16
    patch_n_sigma=1
    compute_grid_shape=(1,256,256)
    Interpolate='nearest'
    grid_plus=False # True
    EPS_plus=False
    EPS_gray=False
    EPS_grid_plus=False
    EPS_edge_plus=False
    dilation=[(3,3),(5,5),(7,7),(9,9)]
    first_input=True

    SMPL_edge=True #True
    SMPL_single=False
    SMPL_Multi=False
    SMPL_overlap=True #True
    smpl_edge_test=False
    # putting your path ex) "/home/user/SEFD/dataset"

    edge_module=False
    edge_module_path='putting your path/SEFD/checkpoint_epoch64.pth'
    distillation_module=True # True
    distillation_module_path="putting your path/SEFD/smpl_overlap.pth.tar"
    distillation_loss="KD_loss" # "L1" or "KD_loss" or "KL_loss" or "ATLoss" or "PKTLoss"
    dilstillation_edge="Canny" # "Canny", "Hed", "PiDiNet", "Edge_Module", "RCF", "edge_module_seg"
    RCF_path="putting your path/SEFD/main/RCFcheckpoint_epoch12.pth"
    distillation_pretrained=True # True
    nothing=False

    inference_testing=False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    # hongsuk choi style
    # KST = datetime.timezone(datetime.timedelta(hours=9))
    # save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
    save_folder = 'exp_' + str(datetime.datetime.now())[5:-10]
    save_folder = save_folder.replace(" ", "_")
    output_dir = osp.join(output_dir, save_folder)
    print('output dir: ', output_dir)
    feature_vis=False

    model_dir = osp.join(output_dir, 'checkpoint')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    def set_args(self, gpu_ids, continue_train=False, is_test=False, exp_dir=''):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.bbox_3d_size = 2
        self.camera_3d_size = 2.5

        if not is_test:
            self.continue_train = continue_train
            if self.continue_train:
                if exp_dir:
                    checkpoints = sorted(glob.glob(osp.join(exp_dir, 'checkpoint') + '/*.pth.tar'), key=lambda x: int(x.split('_')[-1][:-8]))
                    shutil.copy(checkpoints[-1], osp.join(cfg.model_dir, checkpoints[-1].split('/')[-1]))

                else:
                    shutil.copy(osp.join(cfg.root_dir, 'tool', 'snapshot_0.pth.tar'), osp.join(cfg.model_dir, 'snapshot_0.pth.tar'))
        elif is_test and exp_dir:
            self.output_dir = exp_dir
            self.model_dir = osp.join(self.output_dir, 'checkpoint')
            self.vis_dir = osp.join(self.output_dir, 'vis')
            self.log_dir = osp.join(self.output_dir, 'log')
            self.result_dir = osp.join(self.output_dir, 'result')

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        
        if self.testset == 'FreiHAND':
            assert self.trainset_3d[0] == 'FreiHAND'
            assert len(self.trainset_3d) == 1
            assert len(self.trainset_2d) == 0

    def update(self, config_file):
        with open(config_file) as f:
            exp_config = edict(yaml.load(f))
            for k, v in exp_config.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise ValueError("{} not exist in config.py".format(k))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
dataset_list = ['PW3D']
for i in range(len(dataset_list)):
    add_pypath(osp.join(cfg.data_dir, dataset_list[i]))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)