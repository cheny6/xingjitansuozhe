# 环境配置
**1.包**
pip install scikit-learn scikit-optimize pyyaml easydict tqdm

# 数据处理
**1. 放置**
把数据test_a_joint等数据放在ICMEW2024-Track10/Process_data/data目录下；
把数据test_a_joint等数据放在TE-GCN-main/data目录下；
把数据test_a_joint等数据放在A/data目录下；
**2. 生成V2**
```
cd Process_data
python extract_2dpose.py
```
save_2d_pose文件夹下会生成v2，这里并不需要生成3D
**3. Estimate 3d pose 
把所需要的test_a_joint等数据放在ICMEW2024-Track10/Model_inference/Mix_GCN/dataset/save_3d_pose目录下（3d的config不一样）

# 
## Mix_GCN
**CSv2:**
```
python main.py --config ./config/ctrgcn_V2_J.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_B.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_JM.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_BM.yaml --phase train --save-score True --device 0
#######
python main.py --config ./config/ctrgcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_JM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn_V2_BM_3d.yaml --phase train --save-score True --device 0
###
python main.py --config ./config/tdgcn_V2_J.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn_V2_B.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn_V2_JM.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn_V2_BM.yaml --phase train --save-score True --device 0
###
python main.py --config ./config/mstgcn_V2_J.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn_V2_B.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn_V2_JM.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn_V2_BM.yaml --phase train --save-score True --device 0
`
#
## Run Mix_Former
复制数据从**Process_data/save_2d_pose**到**Model_inference/Mix_Former/dataset**目录下
```
cd ./Model_inference/Mix_Former
```
**CSv2:** <br />
```
python main.py --config ./config/mixformer_V2_J.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_B.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_JM.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_BM.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_k2.yaml --phase train --save-score True --device 0 
python main.py --config ./config/mixformer_V2_k2M.yaml --phase train --save-score True --device 0 
```
#
## TEGCN
生成tegcn
cd /doc/TE-GCN-main/model
python main.py --config ./config/uav-cross-subjectv2/train.yaml --work-dir work_dir/2101 -model_saved_name runs/2101 --device 0 --batch-size 56 --test-batch-size 56 --warm_up_epoch 5 --only_train_epoch 60 --seed 777

#
##A
在A里面保存比赛数据然后运行
python gen_angle_data.py
得到生成用于得到tegcn_A和ctrgcn_A的V2
在运行其他部分得到权重等

#
## Ensemble
**参数搜索**
将较高的epoch的pkl和test_label放置在同一个文件夹(该文件夹里面有ensembl.py)
运行ensemble.py
'''
python ensemble.py
'''
最后获得结果

