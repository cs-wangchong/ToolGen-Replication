#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python script/eval_coder.py --dir=models/codet5p-220m-py/lsp-240701-111433 --batch=64