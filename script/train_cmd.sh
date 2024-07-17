#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python script/train_model.py --model=codet5 --lr=5e-6 --epoch=10 --max_len=256 --train_batch=64 --valid_batch=64