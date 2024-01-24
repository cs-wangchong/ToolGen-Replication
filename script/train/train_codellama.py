
import datetime
import logging
from pathlib import Path
import pickle
import time
import json
import random
import torch

from coder.model import init_codellama
from coder.trainer import train
from coder.utils.log_utils import init_log
from coder.constants import PLM_LSP_POINT


if __name__ == '__main__':
    MODEL = "codellama/CodeLlama-7b-Python-hf"

    # LSP_MODE = "nolsp"
    LSP_MODE = "lsp"

    TIMESTAMP = time.strftime("%y%m%d-%H%M%S", time.localtime())
    DIR = f"models-{LSP_MODE}/{MODEL.split('/')[-1]}-{str(TIMESTAMP)}"
    Path(DIR).mkdir(parents=True, exist_ok=True)
    init_log(f"{DIR}/training.log")

    if LSP_MODE == "nolsp":
        train_file, valid_file = "data/datasets/train.bin", "data/datasets/valid-sample3000.bin"
    else:
        train_file, valid_file = "data/datasets/augmented-train.bin", "data/datasets/augmented-valid-sample3000.bin"
    
    logging.info(f"datasets: {(train_file, valid_file)}")

    model, tokenizer, build_prompt, extract_features = init_codellama(
        model_name=MODEL, 
        checkpoint=None,
        additional_tokens=[PLM_LSP_POINT],
        device="cuda"
    )

    with Path(train_file).open("rb") as f:
        train_examples = pickle.load(f)
    with Path(valid_file).open("rb") as f:
        valid_examples = pickle.load(f)
    
    train(
        model,
        tokenizer,
        extract_features,
        train_examples,
        valid_examples, 
        epoch_num=2,
        learning_rate=5e-6,
        weight_decay=0.0,
        train_batch_size=32,
        valid_batch_size=32,
        warmup_steps=200,
        save_limit=3,
        max_len=192,
        save_dir=DIR,
        logging_steps=50,
        valid_steps=500,
    )
