## Running SEFD with various feature connections
In this repository, we provide training and testing codes for SEFD with various edge  
Download the [edge of weights](https://drive.google.com/drive/folders/1lPA0ephyNr0U8b_hqrhLFx_wy9iB19TS?usp=sharing) and place it under `${ROOT}/feat1234` and `${ROOT}/feat234` ... .   
If you want to check the pre-trained weight test, go straight to chapter "Test SEFD with various feature connections"

## Training with various feature connections
❗We compared feature connections only with the Canny edge.❗  
If you want to experiment with a different structure map, simply change the distillation_loss in `${ROOT}/yaml_training/3dpw_{your_structure_map}_distill.yml`. ex) `'log_softmax_l1' => 'ATLoss'`  

* Training-Code (basic feature connection is feat34)
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_"{your_structure_map}"_distil_loss_"{choose_your_loss}".yml
```    
* example (change structure map & loss)
If you use HED as a structure map and Loss as ATLoss, copy and paste `${ROOT}/yaml_training/3dpw_hed_distill.yml => ${ROOT}/yaml_training/3dpw_hed_distill_loss_AT.yml` and line 17: `distillation_loss: "ATLoss"`
```bash  
CUDA_VISIBLE_DEVICES=0 python train.py --amp --continue --gpu 0 --cfg ../assets/yaml_training/3dpw_hed_distil_loss_AT.yml
```    

* example (change feature connections log softmax l1 loss)
If you change feature connection; ex) feat34 => feat 234 check it `${ROOT}/main/model.py`  
model.py => line in 554~566
```bash
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
loss['logsoftmax_feature_4'] = self.l1_loss(b_feature_4,feature4)
```
You can get rid of the # you want. feat34 => feat234
```bash
#b_feature_1=F.log_softmax(b_feature_1,dim=1)
#feature1=F.log_softmax(feature1,dim=1)
b_feature_2=F.log_softmax(b_feature_2,dim=1)
feature2=F.log_softmax(feature2,dim=1)
b_feature_3=F.log_softmax(b_feature_3,dim=1)
feature3=F.log_softmax(feature3,dim=1)
b_feature_4=F.log_softmax(pose_guided_img_feat,dim=1)
feature4=F.log_softmax(feature4,dim=1)
#loss['kd_feature_loss_1'] = self.l1_loss(b_feature_1,feature1)
loss['kd_feature_loss_2'] = self.l1_loss(b_feature_2,feature2)
loss['logsoftmax_feature_3'] = self.l1_loss(b_feature_3,feature3)
loss['logsoftmax_feature_4'] = self.l1_loss(b_feature_4,feature4)
```


* example (change feature connections ATloss)
If you change feature connection; ex) feat1234 => feat 234 check it `${ROOT}/main/model.py`  
model.py => line in 551~553
```bash
elif cfg.distillation_loss=="ATLoss":
    atl=self.atloss([b_feature_1,b_feature_2,b_feature_3,pose_guided_img_feat],[feature1,feature2,feature3,feature4])
    loss['ATLoss']=atl
```
You can get rid of the pose_guided_img_feat & feature4 you want. feat1234 => feat234
```bash
elif cfg.distillation_loss=="ATLoss":
    atl=self.atloss([b_feature_1,b_feature_2,b_feature_3],[feature1,feature2,feature3])
    loss['ATLoss']=atl
```

* 👍 Other losses can be deleted in the same framework. 👍

### Test SEFD with various feature connections
* Pre-trained model testing (Only Canny)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_canny_distil_test.yml --exp_dir ../"{feature_connection}" --test_epoch 10  
```  
* model testing (You should use the structural map you trained.)
```bash  
CUDA_VISIBLE_DEVICES=0 python test.py --amp --continue --gpu 0 --cfg ../assets/yaml_test/3dpw_"{your_strcuture_map}"_distil_test.yml --exp_dir ../output/"{your_Exp_dir}" --test_epoch "your_epoch"  
```  