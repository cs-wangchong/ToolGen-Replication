
import datetime
import logging
from pathlib import Path
import pickle
import time
import json

from coder.utils.log_utils import init_log
from coder.model import init_codellama
from coder.generator import Generator
from coder.evaluator import evaluate, compute_metrics
from coder.constants import *


if __name__ == '__main__':
    MODEL = "codellama/CodeLlama-7b-Python-hf"
    DIR = f"models-nolsp/CodeLlama-7b-Python-hf-231101-123700"
    # TIMESTAMP = time.strftime("%y%m%d-%H%M%S", time.localtime())
    TIMESTAMP = "231103-152153"

    LSP_THRESHOLD = 0.8
    TOKEN_THRESHOLD = 0.0
    TOKEN_K = None
    TEMPERATURE = 1
    MAX_LEN = 156
    REPETITION_PENALTY = 1

    CHECKPOINT = list(sorted(str(ckpt_dir) for ckpt_dir in Path(DIR).glob("checkpoint-*")))[-1]
    with Path(f"{CHECKPOINT}/trainer_state.json").open("r") as f:
        CHECKPOINT = json.load(f)["best_model_checkpoint"]

   
    init_log(f"{DIR}/testing/{TIMESTAMP}.log.es", logging.INFO)
    RESULT_FILE = f"{DIR}/testing/predictions-{TIMESTAMP}.json"
    RECORD_FILE = f"{DIR}/testing/record-{TIMESTAMP}.txt"
        
    test_file = "data/datasets/lsp-test.bin"
    
    with Path(test_file).open("rb") as f:
        test_examples = pickle.load(f)
    
    logging.info(f"model: {MODEL}")
    logging.info(f"checkpoint: {CHECKPOINT}")
    logging.info(f"dataset: {test_file}")
    logging.info(f"dataset size: {len(test_examples)}")
    logging.info(f"max len: {MAX_LEN}")
    logging.info(f"lsp threshold: {LSP_THRESHOLD}")
    logging.info(f"token threshold: {TOKEN_THRESHOLD}")
    logging.info(f"token k: {TOKEN_K}")
    logging.info(f"temperature: {TEMPERATURE}")
    logging.info(f"repeatition penalty: {REPETITION_PENALTY}")

    model, tokenizer, build_prompt, _ = init_codellama(
        model_name=MODEL,
        checkpoint=CHECKPOINT,
        additional_tokens=[PLM_LSP_POINT],
        device="cuda"
    )

    generator = Generator(model, tokenizer, build_prompt)
    bleu, codebleu, edit_sim, hit_rate, syntax_pass, semantic_pass = evaluate(
        generator,
        test_examples, 
        batch_size=24,
        max_len=MAX_LEN,
        lsp_threshold=LSP_THRESHOLD,
        token_threshold=TOKEN_THRESHOLD,
        token_k=TOKEN_K,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        result_file=RESULT_FILE,
        record_file=RECORD_FILE,
        call_lsp=False
    )


    logging.info(f"[Testing]\nblue-4: {round(bleu, 4)}\ncodebleu: {round(codebleu, 4)}\nedit simiarity: {round(edit_sim, 4)}\nlsp hit rate: {round(hit_rate, 4)}\nlint syntax pass rate: {round(syntax_pass, 4)}\nlint semantic pass rate: {round(semantic_pass, 4)}") 
    