#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging

from transformers import AutoTokenizer

from coder.data_augmentor import DataAugmentor
from coder.utils.log_utils import init_log

if __name__ == '__main__':
    init_log("tmp.log", level=logging.INFO)

    pj_path = "test_projects/you-get-b746ac01c9f39de94cac2d56f665285b0523b974"

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    # env_path = "/opt/miniconda3/envs/coder-lsp/bin/python"

    augmentor = DataAugmentor()
    examples = augmentor.handle_project(pj_path)
    for example in examples:
        print(example["augmented_code"])
        print(tokenizer.tokenize(example["augmented_code"]))