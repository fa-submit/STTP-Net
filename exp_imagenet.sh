#!/bin/sh
python3 imagenet_train.py -a resnet50 --dataset Imagenet-LT --loss_type EBS --data_aug CMeO --epochs 200 --num_classes 1000 --workers 15 --print_freq 50 -b 50 --mixup_prob 1 --gpu 0 --start_data_aug 0 --end_data_aug 20 --lr 0.1 --weighted_alpha 1 --exp imagenet_res50

