## Directory  
Almost all the processes were taken from baseline [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE).  
😃Thank you very much for baseline author😃  
### Root  
The `${ROOT}` is described as below.  
```  
${ROOT}  
|-- assets
|-- common  
|-- data  
|-- demo
|-- main  
|-- output  
|-- tool
```  
* `assets` contains config files to reproduce results and some materials used in this repository.
* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for I2L-MeshNet & 3DCrowdNet.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains the current experiment's log, trained models, visualized outputs, and test result (only for MuPoTS).
* `tool` contains codes for auxiliary tasks.
  
### Data  
You need to follow directory structure of the `data` as below.  
❗ There is a very important point. If you follow the directory structure below, be sure to check [here](../data/check.md)❗

```  
${ROOT}  
|-- data 
|   |-- J_regressor_extra.npy 
|   |-- CrowdPose
|   |   |-- annotations
|   |   |-- images
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
|   |   |-- J_regressor_h36m_correct.npy
|   |   |-- SNPL_edge
|   |-- MuCo  
|   |   |-- data  
|   |   |   |-- augmented_set  
|   |   |   |-- unaugmented_set  
|   |   |   |-- MuCo-3DHP.json
|   |   |   |-- smpl_param.json
|   |   |-- SMPL_overlap_edge
|   |-- MSCOCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017 
|   |   |   |-- SMPL_overlap_edge 
|   |   |-- annotations  
|   |   |-- J_regressor_coco_hip_smpl.npy
|   |-- MPII  
|   |   |-- annotations
|   |   |-- images
|   |   |-- SMPL_overlap_edge
|   |-- 3dpw
|   |   |-- data
|   |   |   |-- 3DPW_latest_train.json
|   |   |   |-- 3DPW_latest_validation.json
|   |   |   |-- 3DPW_latest_test.json
|   |   |   |-- 3DPW_validation_crowd_hhrnet_result.json
|   |   |-- imageFiles
|   |   |-- SMPL_overlap_edge1
```  
* Download `J_regressor_*.npy` [[data](https://drive.google.com/drive/folders/187Azod6z13-dS7W5wHerCTgniHYet-yh?usp=sharing)]
* Download CrowdPose data [[data](https://drive.google.com/drive/folders/1qV5Cx5DJLhJVXlfB0vmQrB3ndJXsTZVM?usp=sharing)]
* Download Human3.6M parsed data and SMPL parameters [[data](https://drive.google.com/drive/folders/1kgVH-GugrLoc9XyvP6nRoaFpw3TmM5xK?usp=sharing)][[SMPL parameters from SMPLify-X](https://drive.google.com/drive/folders/1s-yywb4zF_OOLMmw1rsYh_VZFilgSrrD?usp=sharing)]
* Download MuCo parsed/composited data and SMPL parameters [[data](https://drive.google.com/drive/folders/1yL2ey3aWHJnh8f_nhWP--IyC9krAPsQN?usp=sharing)][[SMPL parameters from SMPLify-X](https://drive.google.com/drive/folders/1_JrrbHZICDTe1lqi8S6D_Y1ObmrerAoU?usp=sharing)] 
* Download MS COCO [[data](https://cocodataset.org/#download)] 
* Download MPII data [[data](https://drive.google.com/drive/folders/1zQZpfNu0s19tA7Z1SmulP1cDaVfNDDd3?usp=sharing)]
* Download 3DPW parsed data [[data](https://drive.google.com/drive/folders/1pT0Ix3FxieEQf0HhEbMN1o-DWRzw2Ugh?usp=sharing)]
* Download MS COCO / MPII / CrowdPose SMPL parameters from [NeuralAnnot](https://github.com/mks0601/NeuralAnnot_RELEASE)
* All annotation files follow [MS COCO format](http://cocodataset.org/#format-data).  
* If you want to add your own dataset, you have to convert it to [MS COCO format](http://cocodataset.org/#format-data).  
* All SMPL overlap edge files follow [SMPL overlap edge](https://drive.google.com/drive/folders/1SNSPRPaxm5VhEA7f_0IDF5kmidYOda1D?usp=sharing).  
  
If you have a problem with 'Download limit' problem when tried to download dataset from google drive link, please try this trick.  
```  
* Go the shared folder, which contains files you want to copy to your drive  
* Select all the files you want to copy  
* In the upper right corner click on three vertical dots and select “make a copy”  
* Then, the file is copied to your personal google drive account. You can download it from your personal account.  
```  


### Pytorch SMPL layer and VPoser
* For the SMPL layer, I used [smplpytorch](https://github.com/gulvarol/smplpytorch). The repo is already included in `common/utils/smplpytorch`.
* Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/download.php) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `${ROOT}/smplpytorch/smplpytorch/native/models`.
* Download [VPoser](https://github.com/nghorbani/human_body_prior) from [here](https://drive.google.com/drive/folders/1KNw99d4-_6DqYXfBp2S3_4OMQ_nMW0uQ?usp=sharing) and place it under `${ROOT}/common/utils/human_model_files/smpl/VPOSERR_CKPT`.

### Output  
* Create `output` folder as a soft link form (recommended) instead of a folder form because it would take large storage capacity.  
* The experiments' directory structure will be created as below.
```  
${ROOT}  
|-- output  
|   |-- ${currrent_experiment_name} 
|   |   |-- log  
|   |   |-- checkpoint 
|   |   |-- result  
|   |   |-- vis  
```  
* `log` folder contains training log file.  
* `checkpoint` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files of MuPoTs generated in the testing stage.  
* `vis` folder contains visualized results.  
