## Running SEFD with various edges
In this repository, we provide training and testing codes for SEFD with various edge  
Download the [edge of weights](https://drive.google.com/drive/folders/1cj9U7Jq2B_aN7XagiYmkt_o10B8rzF4p?usp=sharing)  

### Setting the position of edge weight
We preset the locations of PiDiNet, RCF, HED, edge estimator, and SMPL overlap edge.
* PiDiNet place it under `${ROOT}/table5_pidinet.pth`.  
* RCF place it under `${ROOT}/main/RCFcheckpoint_epoch12.pth`.  
* The HED automatically downloads the pth file.  
* edge estimator place it under `${ROOT}/checkpoint_epoch64.pth`. 
* SMPL overlap edge place it under `${ROOT}/smpl_overlap.pth.tar`.

### Train & Test without using feature distillation  
❗This code does not use feature distillation.❗  
* training Structure map (PiDiNet)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_pidinet_nodistil.yml
```  
* training Structure map (RCF)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_RCF_nodistil.yml
```  
* training Structure map (HED)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_hed_nodistil.yml
```  
* training Structure map (Canny)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_canny_nodistil.yml
```  
* training Structure map (SEE)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_SEE_nodistil.yml
```  

* testing Structure map (PiDiNet)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_pidinet_nodistil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (RCF)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_RCF_nodistil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (HED)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_HED_nodistil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (Canny)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_canny_nodistil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  

### Train & Test with using feature distillation  
This code use feature distillation. 
* training Structure map (PiDiNet)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_pidinet_distil.yml
```  
* training Structure map (RCF)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_RCF_distil.yml
```  
* training Structure map (HED)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_hed_distil.yml
```  
* training Structure map (Canny)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_canny_distil.yml
```  
* training Structure map (SEE)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_SEE_distil.yml
```  

* testing Structure map (PiDiNet)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_pidinet_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (RCF)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_RCF_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (HED)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_HED_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (Canny)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_canny_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* testing Structure map (SEE)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_SEE_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  

## Test MuPoTs
* MuPoTs_test_openpose_result.json & MuPoTs_test_hhrnet_result.json place it under `${ROOT}/tool/OO.json`.
* You should create a folder called "MuPoTs" in the `"${ROOT}/output/{your_Exp_dir}/result"` to test. like this `"${ROOT}/output/{your_Exp_dir}/result/MuPoTs"`
* `${ROOT}/asset/yaml_test/3dpw_{structure map}_{distill or nodistil}_test.yml` Change the testset to "MuPoTs" in the yaml file of the structural map to be tested at the location
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_"{structure map}"_"{distil or nodisitil}"_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "{your_epoch}"  
```  
* Finally, you can use the matlab evaluation code shown in [this](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). If you want to run it as python, you can check [this](https://github.com/ddddwee1/MuPoTS3D-Evaluation).