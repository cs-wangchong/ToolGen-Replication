
import datetime
import logging
from pathlib import Path
import pickle
import time
import json
import argparse
import os

from coder.utils.log_utils import init_log
from coder.model import *
from coder.generator import Generator
from coder.evaluator import evaluate
from coder.constants import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_FACTORY = {
    "codegen": ("Salesforce/codegen-350M-mono", init_codegen, 2048),
    "deepseek": ("deepseek-ai/deepseek-coder-1.3b-base", init_deepseek, 16384),
    "codellama": ("codellama/CodeLlama-7b-Python-hf", init_codellama, 4096),
    "deepseek-lora": ("deepseek-ai/deepseek-coder-1.3b-base", init_deepseek_lora, 16384),
    "codellama-lora": ("codellama/CodeLlama-7b-Python-hf", init_codellama_lora, 4096),
    "codet5": ("Salesforce/codet5p-220m-py", init_codet5, 512),
    "codegpt": ("microsoft/CodeGPT-small-py", init_codegpt, 1024),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-model", "--model", required=True, type=str)
    parser.add_argument("-dir", "--dir", required=False, type=str, default=None)
    # parser.add_argument("-stamp", "--timestamp", required=False, type=str, default=None)
    # parser.add_argument("-lsp", "--lsp", required=False, default=False, action="store_true")
    parser.add_argument("-sample", "--sample", required=False, default=False, action="store_true")
    
    parser.add_argument("-lsp_conf", "--lsp_conf", required=False, type=float, default=0.5)
    parser.add_argument("-token_conf", "--token_conf", required=False, type=float, default=0.0)
    parser.add_argument("-token_k", "--token_k", required=False, type=float, default=None)
    parser.add_argument("-temp", "--temp", required=False, type=float, default=1.0)
    parser.add_argument("-max_len", "--max_len", required=False, type=int, default=192)
    parser.add_argument("-batch", "--batch", required=False, type=int, default=24)
    args = parser.parse_args()

    DIR = args.dir
    SAMPLE = args.sample
    # TIMESTAMP = args.timestamp if args.timestamp else time.strftime("%y%m%d-%H%M%S", time.localtime())
    LSP_CONF = args.lsp_conf
    TOKEN_THRESHOLD = args.token_conf
    TOKEN_K = args.token_k
    TEMPERATURE = args.temp
    MAX_LEN = args.max_len
    REPETITION_PENALTY = 1
    BATCH_SIZE = args.batch
    DEVICE = "cuda"


    with Path(f"{DIR}/meta.json").open("r") as f:
        meta = json.load(f)
        MODEL_NAME, INIT_FUNC, MODEL_MAX_LEN = MODEL_FACTORY[meta["model"]]
        CALL_LSP = meta["lsp"]

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
        init_log(f"{DIR}/testing/testing.log", logging.INFO)
        PRED_RESULT_FILE = f"{DIR}/testing/predictions.json"
        LINT_RESULT_FILE = f"{DIR}/testing/lintresults.json"
        RECORD_FILE = f"{DIR}/testing/record.txt"
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

    model, tokenizer = INIT_FUNC(
        model_name=MODEL_NAME,
        checkpoint=CHECKPOINT,
        additional_tokens=[PLM_LSP_POINT] if CALL_LSP else [],
        device=DEVICE
    )

    generator = Generator(model, tokenizer, MODEL_MAX_LEN)

    exact, bleu, codebleu, edit_sim, hit_rate, syntax_pass, semantic_pass = evaluate(
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
    logging.info(f"[Testing]\nexact match: {round(exact, 4)}\nblue-4: {round(bleu, 4)}\ncodebleu: {round(codebleu, 4)}\nedit simiarity: {round(edit_sim, 4)}\nlsp hit rate: {round(hit_rate, 4)}\nlint syntax pass rate: {round(syntax_pass, 4)}\nlint semantic pass rate: {round(semantic_pass, 4)}") 