
###
 # @Date: 2023-10-18 22:12:25
 # @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 # @LastEditTime: 2023-10-18 23:42:34
 # @FilePath: /CAC4SSL/test.sh
### 
gpuid=1

# python ./code/train_3d.py --dataset_name LA --model cacnet3d_emb_64 --labelnum 4 --gpu $gpuid --temperature 0.1 --exp cacnet_sample_50 && \
# python ./code/train_3d.py --dataset_name LA --model cacnet3d_emb_64 --labelnum 8 --gpu $gpuid --temperature 0.1 --exp cacnet_sample_50 && \
# python ./code/train_3d.py --dataset_name LA --model cacnet3d_emb_64 --labelnum 16 --gpu $gpuid --temperature 0.1 --exp cacnet_sample_50 && \

# python ./code/train_3d.py --dataset_name Pancreas_CT --model cacnet3d_emb_64 --labelnum 6 --gpu $gpuid --temperature 0.1 --exp cacnet_sample_50 && \
# python ./code/train_3d.py --dataset_name Pancreas_CT --model cacnet3d_emb_64 --labelnum 12 --gpu $gpuid --temperature 0.1 --exp cacnet_sample_50 && \

# python ./code/test_3d.py --dataset_name LA --model cacnet3d_emb_64 --exp cacnet_sample_50 --labelnum 4 --gpu $gpuid && \
# python ./code/test_3d.py --dataset_name LA --model cacnet3d_emb_64 --exp cacnet_sample_50 --labelnum 8 --gpu $gpuid && \
# python ./code/test_3d.py --dataset_name LA --model cacnet3d_emb_64 --exp cacnet_sample_50 --labelnum 16 --gpu $gpuid && \

# python ./code/test_3d.py --dataset_name Pancreas_CT --model cacnet3d_emb_64 --exp cacnet_sample_50 --labelnum 6 --gpu $gpuid && \
# python ./code/test_3d.py --dataset_name Pancreas_CT --model cacnet3d_emb_64 --exp cacnet_sample_50 --labelnum 12 --gpu $gpuid && \


exit 0;
