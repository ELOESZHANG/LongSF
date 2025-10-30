#!/usr/bin/env bash

#creat kitti_pkl and gt
#python -m pcdet.datasets.kitti.kitti_dataset_custom create_kitti_infos ../tools/cfgs/dataset_configs/kitti_dataset_custom.yaml

##Train
##############  LongsF: 1 GPU ######################
CUDA_VISIBLE_DEVICES='0' python train.py --gpu_id 1 --workers 1 --cfg_file cfgs/kitti_models/longsf.yaml \
  --batch_size 1 --epochs 60 --max_ckpt_save_num 20  \
  --fix_random_seed


##Val all.  You need to create a soft link from longsf/default/ckpt to longsf_test/default/ckpt.
#CUDA_VISIBLE_DEVICES='0' python test.py --gpu_id 1 --workers 4 --cfg_file cfgs/kitti_models/longsf_test.yaml --batch_size 1 \
#  --eval_all
#
#
##Val one epoch
#CUDA_VISIBLE_DEVICES='0' python test.py --gpu_id 1 --workers 4 --cfg_file cfgs/kitti_models/longsf_test.yaml --batch_size 1 \
#  --ckpt ../output/kitti_models/longsf/default/ckpt/checkpoint_epoch_52.pth  \
#  ###--save_to_file
#
#
###Train with 2 GPUs
#python -m torch.distributed.launch --nnodes 1 --nproc_per_node=2 --master_port 25511 train.py --gpu_id 0 --launch 'pytorch' --workers 4 \
#   --batch_size 2 --cfg_file cfgs/kitti_models/mpcf_can_mamba.yaml  --tcp_port 61000  \
#   --epochs 60 --max_ckpt_save_num 30 \
#   --fix_random_seed \
#
#
#
################## For high performance  ####################
##1. train Baseline. You need annotate the ISF and TSR modules in longsf_part.py. And Set lr=0.001 in longsf.yaml.
#CUDA_VISIBLE_DEVICES='0' python train.py --gpu_id 1 --workers 1 --cfg_file cfgs/kitti_models/longsf.yaml \
#  --batch_size 1 --epochs 60 --max_ckpt_save_num 20  \
#  --fix_random_seed

#2. train longsf. Cancel the annotations of ISF and TSR modules in longsf_part.py. And Set lr=0.00001 in longsf.yaml.
CUDA_VISIBLE_DEVICES='0' python train.py --gpu_id 1 --workers 1 --cfg_file cfgs/kitti_models/longsf.yaml \
  --batch_size 1 --epochs 30 --max_ckpt_save_num 20  \
  --fix_random_seed \
  --pretrained_model ../output/kitti_models/longsf/default/ckpt/checkpoint_epoch_52.pth




