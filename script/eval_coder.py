
import datetime
import logging
from pathlib import Path
import pickle
import time
import json
import argparse

from coder.utils.log_utils import init_log
from coder.model import init_codellama, init_codet5, init_codegpt
from coder.generator import Generator
from coder.evaluator import evaluate
from coder.constants import *


MODEL_FACTORY = {
    "codellama": ("codellama/CodeLlama-7b-Python-hf", init_codellama),
    "codet5": ("Salesforce/codet5p-220m-py", init_codet5),
    "codegpt": ("microsoft/CodeGPT-small-py", init_codegpt),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", required=True, type=str)
    parser.add_argument("-version", "--version", required=False, type=str, default=None)
    parser.add_argument("-stamp", "--timestamp", required=False, type=str, default=None)
    parser.add_argument("-lsp", "--lsp", required=False, default=True, action="store_true")
    parser.add_argument("-sample", "--sample", required=False, default=False, action="store_true")
    parser.add_argument("-gpu", "--gpu", required=False, default=True, action="store_true")
    
    parser.add_argument("-lspconf", "--lsp_confidence", required=False, type=float, default=0.5)
    parser.add_argument("-tkconf", "--token_confidence", required=False, type=float, default=0.0)
    parser.add_argument("-tknum", "--token_number", required=False, type=float, default=None)
    parser.add_argument("-temp", "--temperature", required=False, type=float, default=1.0)
    parser.add_argument("-maxlen", "--max_len", required=False, type=int, default=160)
    parser.add_argument("-batchsize", "--batch_size", required=False, type=int, default=24)
    args = parser.parse_args()

    MODEL_NAME, INIT_FUNC = MODEL_FACTORY[args.model]
    CALL_LSP = args.lsp
    SAMPLE = args.sample
    DIR = args.version
    TIMESTAMP = args.timestamp if args.timestamp else time.strftime("%y%m%d-%H%M%S", time.localtime())
    LSP_CONF = args.lsp_confidence
    TOKEN_THRESHOLD = args.token_confidence
    TOKEN_K = args.token_number
    TEMPERATURE = args.temperature
    MAX_LEN = args.max_len
    REPETITION_PENALTY = 1

    BATCH_SIZE = args.batch_size
    DEVICE = "cuda" if args.gpu else "cpu"

    CHECKPOINT = list(sorted(str(ckpt_dir) for ckpt_dir in Path(DIR).glob("checkpoint-*")))[-1]
    with Path(f"{CHECKPOINT}/trainer_state.json").open("r") as f:
        CHECKPOINT = json.load(f)["best_model_checkpoint"]

    
    if SAMPLE:
        init_log(f"{DIR}/testing-sampled/testing.log", logging.INFO)
        if CALL_LSP:
            PRED_RESULT_FILE = f"{DIR}/testing-sampled/predictions-calllsp.json"
            LINT_RESULT_FILE = f"{DIR}/testing-sampled/lintresults-calllsp.json"
            RECORD_FILE = f"{DIR}/testing-sampled/record-calllsp.txt"
        else:
            PRED_RESULT_FILE = f"{DIR}/testing-sampled/predictions-direct.json"
            LINT_RESULT_FILE = f"{DIR}/testing-sampled/lintresults-direct.json"
            RECORD_FILE = f"{DIR}/testing-sampled/record-direct.txt"
        test_file = "data/datasets/lsp-test-sample100.bin"
    else:
        init_log(f"{DIR}/testing/{TIMESTAMP}.log", logging.INFO)
        PRED_RESULT_FILE = f"{DIR}/testing/predictions-{TIMESTAMP}.json"
        LINT_RESULT_FILE = f"{DIR}/testing/lintresults-{TIMESTAMP}.json"
        RECORD_FILE = f"{DIR}/testing/record-{TIMESTAMP}.txt"
        test_file = "data/datasets/lsp-test.bin"
        
    with Path(test_file).open("rb") as f:
        test_examples = pickle.load(f)
    
    logging.info(f"model: {MODEL_NAME}")
    logging.info(f"checkpoint: {CHECKPOINT}")
    logging.info(f"dataset: {test_file}")
    logging.info(f"dataset size: {len(test_examples)}")
    logging.info(f"max len: {MAX_LEN}")
    logging.info(f"lsp confidence threshold: {LSP_CONF}")
    logging.info(f"repeatition penalty: {REPETITION_PENALTY}")

    model, tokenizer, build_prompt, _ = INIT_FUNC(
        model_name=MODEL_NAME,
        checkpoint=CHECKPOINT,
        additional_tokens=[PLM_LSP_POINT],
        device=DEVICE
    )

    generator = Generator(model, tokenizer, build_prompt)

    bleu, codebleu, edit_sim, hit_rate, syntax_pass, semantic_pass = evaluate(
        generator,
        test_examples, 
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        lsp_conf=LSP_CONF,
        # token_threshold=TOKEN_THRESHOLD,
        # token_k=TOKEN_K,
        # temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        pred_result_file=PRED_RESULT_FILE,
        lint_result_file=LINT_RESULT_FILE,
        record_file=RECORD_FILE,
        call_lsp=CALL_LSP
    )
    logging.info(f"[Testing]\nblue-4: {round(bleu, 4)}\ncodebleu: {round(codebleu, 4)}\nedit simiarity: {round(edit_sim, 4)}\nlsp hit rate: {round(hit_rate, 4)}\nlint syntax pass rate: {round(syntax_pass, 4)}\nlint semantic pass rate: {round(semantic_pass, 4)}") 
