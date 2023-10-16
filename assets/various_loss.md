## Running SEFD with various losses
In this repository, we provide training and testing codes for SEFD with various loss  
Download the [edge of weights](https://drive.google.com/drive/folders/1cj9U7Jq2B_aN7XagiYmkt_o10B8rzF4p?usp=sharing) and place it under `${ROOT}/Log_softmax` and `${ROOT}/L1_loss` ... .  
If you want to check the pre-trained weight test, go straight to chapter "Test SEFD with various losses"

## Training with various losses
❗We compared loss only with the Canny edge.❗  
If you want to experiment with a different structure map, simply change the distillation_loss in `${ROOT}/yaml_training/3dpw_{your_structure_map}_distill.yml`. ex) `'log_softmax_l1' => 'ATLoss'`  

* Training-Code
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_canny_distil_loss_"{choose_your_loss}".yml
```    
* example
If you use HED as a structure map and Loss as ATLoss, copy and paste `${ROOT}/yaml_training/3dpw_hed_distill.yml => ${ROOT}/yaml_training/3dpw_hed_distill_loss_AT.yml` and line 17: `distillation_loss: "ATLoss"`
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_hed_distil_loss_AT.yml
```    


### Test with various losses
* Pre-trained model testing (Only Canny)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_canny_distil_test.yml --exp_dir ../"{choose_loss}" --test_epoch 10  
```  
* model testing (You should use the structural map you trained.)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_"{your_strcuture_map}"_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "your_epoch"  
```  