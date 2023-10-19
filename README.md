
# Contour-Aware Consistency for Semi-Supervised Medical Image Segmentation
by Lei Li*, Sheng Lian, Zhiming Luo, Beizhan Wang, Shaozi Li

### Introduction
This repository is for our paper: '[Contour-Aware Consistency for Semi-Supervised Medical Image Segmentation]()'. 

### Requirements
This repository is based on PyTorch 1.8.0, CUDA 11.1 and Python 3.8.0; All experiments in our paper were conducted on a single NVIDIA GeForce RTX 3090 GPU.

### Usage
1. Clone the repo.;
```
git clone https://github.com/SmileJET/CAC4SSL.git
```
2. Put the data in './CAC4SSL/data';

3. Train the model;
```
cd CAC4SSL
# e.g., for 20% labels on LA
python ./code/train_3d.py --dataset_name LA --model cacnet3d_emb_64 --labelnum 16 --gpu 0 --temperature 0.1 --exp cacnet_sample_50

```
4. Test the model;
```
cd CAC4SSL
# e.g., for 20% labels on LA
python ./code/test_3d.py --dataset_name LA --model cacnet3d_emb_64 --exp cacnet_sample_50 --labelnum 16 --gpu 0
```


### Acknowledgements:
Our code is origin from [MC-Net](https://github.com/ycwu1997/MC-Net), [UAMT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [DTC](https://github.com/HiLab-git/DTC), [URPC](https://github.com/HiLab-git/SSL4MIS) and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.
