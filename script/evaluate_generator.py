
import datetime
import logging
from pathlib import Path
import pickle
import time
import random

from coder.utils.log_utils import init_log
from coder.generator import Generator


if __name__ == '__main__':
    BEAM = 1
    MAX_LEN = 128
    REPETITION_PENALTY = 2.5

    TIMESTAMP = time.strftime("%y%m%d-%H%M%S", time.localtime())
    DIR = f"models-nolsp/codet5p-220m-py-231024-174959"

    init_log(f"{DIR}/testing/beam{BEAM}.log", logging.INFO)
    RECORD_FILE = f"{DIR}/testing/record-beam{BEAM}.txt"
    
    test_file = "data/datasets/lsp-test.bin"
    with Path(test_file).open("rb") as f:
        test_examples = pickle.load(f)
    test_examples = [(prompt, code) for (_, _, prompt, code) in test_examples]
    
    logging.info(f"dataset: {test_file}")
    logging.info(f"dataset size: {len(test_examples)}")
    logging.info(f"beam size: {BEAM}")
    logging.info(f"max len: {MAX_LEN}")
    logging.info(f"repeatition penalty: {REPETITION_PENALTY}")
    
    generator = Generator.load_from_model_info(f"{DIR}/model.json", device="cuda")
    blue, codebleu, lsp_hit = generator.evaluate(
        test_examples, 
        batch_size=64,
        max_len=MAX_LEN,
        beam_size=BEAM,
        repetition_penalty=REPETITION_PENALTY,
        record_file=RECORD_FILE
    )
    logging.info(f"[Testing] blue-4: {round(blue, 4)}, codebleu: {round(codebleu, 4)}, lsp hit: {round(lsp_hit, 4)}") 