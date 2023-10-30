
import datetime
import logging
from pathlib import Path
import pickle
import time
import json
import random

from coder.generator import Generator
from coder.utils.log_utils import init_log
from coder.constants import PLM_LSP_POINT


if __name__ == '__main__':
    MODEL = "Salesforce/codet5p-220m-py"

    # LSP_MODE = "nolsp"
    LSP_MODE = "lsp"

    TIMESTAMP = time.strftime("%y%m%d-%H%M%S", time.localtime())
    DIR = f"models-{LSP_MODE}/{MODEL.split('/')[-1]}-{str(TIMESTAMP)}"
    Path(DIR).mkdir(parents=True, exist_ok=True)
    init_log(f"{DIR}/training.log")

    if LSP_MODE == "nolsp":
        train_file, valid_file, test_file = "data/datasets/train.bin", "data/datasets/valid.bin", "data/datasets/test.bin"
    else:
        train_file, valid_file, test_file = "data/datasets/augmented-train.bin", "data/datasets/augmented-valid.bin", "data/datasets/augmented-test.bin"
    
    logging.info(f"datasets: {(train_file, valid_file, test_file)}")

    generator = Generator(
        MODEL,
        additional_tokens=[PLM_LSP_POINT],
        device="cuda", 
    )

    with Path(train_file).open("rb") as f:
        train_examples = pickle.load(f)
    with Path(valid_file).open("rb") as f:
        valid_examples = pickle.load(f)
    
    generator.train(
        train_examples,
        valid_examples, 

        epoch_num=10,
        learning_rate=1e-5,
        train_batch_size=32,
        valid_batch_size=48,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup_steps=0.1,
        max_grad_norm=1.0,
        train_sampling=None,
        valid_sampling=None,
        best_k=3,
        patience=5,
        max_len=256,
        save_dir=DIR,
        log_step=500,
        valid_step=None,
        report_prediction=True,
    )

    init_log(f"{DIR}/testing.log")
    test_file = "data/datasets/lsp-test.bin"
    with Path(test_file).open("rb") as f:
        test_examples = pickle.load(f)
    test_examples = [(prompt, code) for (_, _, prompt, code) in test_examples]
    # test_examples = random.sample(test_examples, 10)
    blue, codebleu, lsp_hit = generator.evaluate(test_examples, batch_size=48, max_len=256, record_file=f"{DIR}/record-test.txt")
    logging.info(f"[Testing] blue-4: {round(blue, 4)}, codebleu: {round(codebleu, 4)}, lsp hit: {round(lsp_hit, 4)}") 
    