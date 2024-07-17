import logging
from pathlib import Path
import pickle
import time
import json
import argparse
import os

from coder.model import *
from coder.trainer import train
from coder.utils.log_utils import init_log
from coder.constants import PLM_LSP_POINT

os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_FACTORY = {
    "codegen": ("Salesforce/codegen-350M-mono", init_codegen),
    "deepseek": ("deepseek-ai/deepseek-coder-1.3b-base", init_deepseek),
    "codellama": ("codellama/CodeLlama-7b-Python-hf", init_codellama),
    "deepseek-lora": ("deepseek-ai/deepseek-coder-1.3b-base", init_deepseek_lora),
    "codellama-lora": ("codellama/CodeLlama-7b-Python-hf", init_codellama_lora),
    "codet5": ("Salesforce/codet5p-220m-py", init_codet5),
    "codegpt": ("microsoft/CodeGPT-small-py", init_codegpt),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", required=True, type=str)
    parser.add_argument("-lsp", "--lsp", required=False, default=False, action="store_true")
    parser.add_argument("-lr", "--lr", required=False, type=float, default=1e-5)
    parser.add_argument("-epoch", "--epoch", required=False, type=int, default=3)
    parser.add_argument("-max_len", "--max_len", required=False, type=int, default=256)
    parser.add_argument("-train_batch", "--train_batch", required=False, type=int, default=32)
    parser.add_argument("-valid_batch", "--valid_batch", required=False, type=int, default=32)

    args = parser.parse_args()

    MODEL_NAME, INIT_FUNC = MODEL_FACTORY[args.model]
    LSP_MODE = "lsp" if args.lsp else "nolsp"

    TIMESTAMP = time.strftime("%y%m%d-%H%M%S", time.localtime())
    DIR = f"models/{MODEL_NAME.split('/')[-1]}/{LSP_MODE}-{str(TIMESTAMP)}"
    Path(DIR).mkdir(parents=True, exist_ok=True)
    with Path(f"{DIR}/meta.json").open("w") as f:
        json.dump({"model": args.model, "lsp": args.lsp}, f, indent=4)
    init_log(f"{DIR}/training.log")

    if LSP_MODE == "nolsp":
        train_file, valid_file = "data/datasets/train.bin", "data/datasets/valid-sample3000.bin"
    else:
        train_file, valid_file = "data/datasets/augmented-train.bin", "data/datasets/augmented-valid-sample3000.bin"
    
    logging.info(f"datasets: {(train_file, valid_file)}")

    model, tokenizer = INIT_FUNC(
        model_name=MODEL_NAME, 
        checkpoint=None,
        additional_tokens=[PLM_LSP_POINT] if LSP_MODE == "lsp" else [],
        device="cuda"
    )

    with Path(train_file).open("rb") as f:
        train_examples = pickle.load(f)
    with Path(valid_file).open("rb") as f:
        valid_examples = pickle.load(f)
    
    train(
        model,
        tokenizer,
        train_examples,
        valid_examples, 
        epoch_num=args.epoch,
        learning_rate=args.lr,
        weight_decay=0.0,
        train_batch_size=args.train_batch,
        valid_batch_size=args.valid_batch,
        warmup_steps=100,
        save_limit=3,
        max_len=args.max_len,
        save_dir=DIR,
        logging_steps=50,
        valid_steps=1000,
    )