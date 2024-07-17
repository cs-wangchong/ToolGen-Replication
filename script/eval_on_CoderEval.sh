#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python script/eval_coder_on_CoderEval.py --batch=1 --dir=models/CodeGPT-small-py/lsp-240701-045716/
CUDA_VISIBLE_DEVICES=0 python script/eval_coder_on_CoderEval.py --batch=1 --dir=models/CodeGPT-small-py/nolsp-240701-045650
CUDA_VISIBLE_DEVICES=0 python script/eval_coder_on_CoderEval.py --batch=1 --dir=models/codet5p-220m-py/lsp-240701-111433
CUDA_VISIBLE_DEVICES=0 python script/eval_coder_on_CoderEval.py --batch=1 --dir=models/codet5p-220m-py/nolsp-240701-111416

