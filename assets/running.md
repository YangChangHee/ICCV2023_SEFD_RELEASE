## Running SEFD
In this repository, we provide training and testing codes for 3DPW (Table 4) and RH-dataset, OCHuman, CrowPose (Table 5).
We use the pre-trained ResNet-50 weights of [xiao2018simple](https://github.com/microsoft/human-pose-estimation.pytorch) to achieve faster convergence, but you can get the same result by training longer.
Download the [file of weights](https://drive.google.com/drive/folders/1UsntO3wdIHOiajcb8oicMhQ82SmFvulp?usp=sharing) and place it under `${ROOT}/tool/`.  

Download the [smpl_overlap_edge.pth.tar](https://drive.google.com/drive/folders/1cj9U7Jq2B_aN7XagiYmkt_o10B8rzF4p?usp=sharing) and place it under `${ROOT}/smpl_overlap_edge.pth.tar`.  

### Test  
Download the experiment directories from [here](https://drive.google.com/drive/folders/1LOfLLCf7_iApeiKMsyAUW5bnJEkBgS3L?usp=sharing) and place them under `${ROOT}/distillation/`.  
To evaluate on 3DPW (Table 6,7,8,9), run 
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --gpu 0 --cfg ../assets/yaml_test/3dpw_distil_test.yml --exp_dir ../distillation/distill_canny --test_epoch 10 
```  

### Test MuPoTs
You can check it out [here](./various_edge.md)  
(Check the last one)