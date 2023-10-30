
import datetime
import logging
from pathlib import Path
import pickle
import time
import random

from coder.utils.log_utils import init_log
from coder.generator import Generator
from coder.lsp_coder import LSPCoder

SAMPLED = False
# SAMPLED = True

if __name__ == '__main__':
    CAND_NUM = 1
    LSP_THRESHOLD = 0.8
    TOKEN_THRESHOLD = 0.0
    TOKEN_K = 10
    TEMPERATURE = 1
    BEAM = 1
    MAX_LEN = 128
    REPETITION_PENALTY = 2.5

    TIMESTAMP = time.strftime("%y%m%d-%H%M%S", time.localtime())
    DIR = f"models-lsp/codet5p-220m-py-231024-175010"

    if SAMPLED:
        init_log(f"{DIR}/testing-sampled/{TIMESTAMP}.log", logging.INFO)
        RECORD_FILE_CALLLSP = f"{DIR}/testing-sampled/record-calllsp-{TIMESTAMP}.txt"
        RECORD_FILE_DERIECT = f"{DIR}/testing-sampled/record-direct-{TIMESTAMP}.txt"
    else:
        init_log(f"{DIR}/testing/{TIMESTAMP}.log", logging.INFO)
        RECORD_FILE_CALLLSP = f"{DIR}/testing/record-{TIMESTAMP}.txt"
        
    
    if SAMPLED:
        test_file = "data/datasets/lsp-test-sample100.bin"
    else:
        test_file = "data/datasets/lsp-test.bin"
    
    with Path(test_file).open("rb") as f:
        test_examples = pickle.load(f)
    
    logging.info(f"dataset: {test_file}")
    logging.info(f"dataset size: {len(test_examples)}")
    logging.info(f"beam size: {BEAM}")
    logging.info(f"max len: {MAX_LEN}")
    logging.info(f"cand num: {CAND_NUM}")
    logging.info(f"lsp threshold: {LSP_THRESHOLD}")
    logging.info(f"token threshold: {TOKEN_THRESHOLD}")
    logging.info(f"token k: {TOKEN_K}")
    logging.info(f"temperature: {TEMPERATURE}")
    logging.info(f"repeatition penalty: {REPETITION_PENALTY}")
    
    generator = Generator.load_from_model_info(f"{DIR}/model.json", device="cuda")
    if SAMPLED:
        blue, codebleu, lsp_hit = generator.evaluate(
            [(prompt, code) for (_, _, prompt, code) in test_examples], 
            batch_size=16,
            max_len=MAX_LEN,
            beam_size=BEAM,
            repetition_penalty=REPETITION_PENALTY,
            record_file=RECORD_FILE_DERIECT
        )
        logging.info(f"[Testing] DERICT, blue-4: {round(blue, 4)}, codebleu: {round(codebleu, 4)}, lsp hit: {round(lsp_hit, 4)}") 
    
    
    lsp_coder = LSPCoder(generator)
    blue, codebleu, lsp_hit = lsp_coder.evaluate_with_lsp(
        test_examples, 
        batch_size=1,
        max_len=MAX_LEN,
        beam_size=BEAM,
        cand_num=CAND_NUM,
        lsp_threshold=LSP_THRESHOLD,
        token_threshold=TOKEN_THRESHOLD,
        token_k=TOKEN_K,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        record_file=RECORD_FILE_CALLLSP
    )
    logging.info(f"[Testing] CALL LSP, blue-4: {round(blue, 4)}, codebleu: {round(codebleu, 4)}, lsp hit: {round(lsp_hit, 4)}") 
    