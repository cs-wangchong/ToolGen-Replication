import json
import logging
import random
from pathlib import Path
import re
import os
import subprocess
import string
from collections import defaultdict
import gc

from tqdm import tqdm
import torch
from coder.generator import Generator, clean_lsp
from coder.data_augmentor import DataAugmentor
from coder.utils.metric_utils import Bleu, CodeBleu
from fuzzywuzzy import fuzz
from coder.constants import *

torch.manual_seed(42)  # pytorch random seed

def clean_str(code):
    code = re.sub(r"'(.*?)'", "''", code)
    code = re.sub(r'"(.*?)"', "''", code)
    return code.strip()
    

def evaluate(
        generator: Generator,
        eval_set,
        batch_size=32,
        max_len=256,
        lsp_conf=0.8,
        repetition_penalty=1.0,
        pred_result_file=None,
        lint_result_file=None,
        record_file=None,
        call_lsp=False
    ):
    if pred_result_file is None or not Path(pred_result_file).exists():
        predictions = []
        if not call_lsp:
            batch_ranges = list(zip(range(0, len(eval_set), batch_size), range(batch_size, len(eval_set)+batch_size, batch_size)))
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Evaluation"):
                batch = [
                    {
                        "path": file_path,
                        "context": context,
                        "line": line,
                        "column": column,
                        "docstr": docstr,
                        "signature": signature,
                        "prefix": prefix,
                    }
                    for _, file_path, context, line, column, docstr, signature, code, prefix, body in eval_set[beg:end]
                ]
                outputs = generator.generate_simple(batch, max_len=max_len, repetition_penalty=repetition_penalty)
                predictions.extend(outputs)
        else:
            repo2insts = defaultdict(list)
            for idx, (repo, file_path, context, line, column, docstr, signature, code, prefix, body) in enumerate(eval_set, 1):
                repo2insts[repo].append((idx, file_path, context, line, column, docstr, signature, code, prefix, body))
            for repo, insts in tqdm(repo2insts.items(), total=len(repo2insts), ascii=True, desc="Evaluation"):
                generator.update_lsp_project(repo)
                batch_ranges = list(zip(range(0, len(insts), batch_size), range(batch_size, len(insts)+batch_size, batch_size)))
                for beg, end in batch_ranges:
                    idxes = [idx for idx, *_ in insts[beg:end]]
                    batch = [
                        {
                            "path": file_path,
                            "context": context,
                            "line": line,
                            "column": column,
                            "docstr": docstr,
                            "signature": signature,
                            "prefix": prefix,
                        }
                        for _, file_path, context, line, column, docstr, signature, code, prefix, body in insts[beg:end]
                    ]
                    # logging.info(f"====================================")
                    # logging.info(f"[EXPECTATION]\n{code.strip()}")
                    outputs = generator.generate_with_lsp(
                        batch,
                        max_len=max_len,
                        lsp_conf=lsp_conf,
                        repetition_penalty=repetition_penalty
                    )
                    predictions.extend(zip(idxes, outputs))
            predictions = [pred for _, pred in sorted(predictions, key=lambda p: p[0])]
        if pred_result_file is not None:
            with Path(pred_result_file).open("w") as f:
                json.dump(predictions, f, indent=4)
    else:
        with Path(pred_result_file).open("r") as f:
            predictions = json.load(f)
    
    inputs, contexts, expectations = [], [], []
    for repo, file_path, context, line, column, docstr, signature, code, prefix, body in eval_set:
        inputs.append((docstr, signature))
        contexts.append((file_path, context, line))
        if generator.model.config.is_encoder_decoder:
            expectations.append(code)
        else:
            expectations.append(f"{prefix}{body}")

    generator.model.to("cpu")
    generator.model = None
    del generator.model
    torch.cuda.empty_cache()
    gc.collect()
    
    return compute_metrics(generator.tokenizer, inputs, contexts, expectations, predictions, lint_result_file, record_file)

def lint(contexts, predictions):
    lint_results = []
    for (file_path, context, line), pred in tqdm(zip(contexts, predictions), desc="Lint", total=len(contexts), ascii=True):
        errors = []
        end_line = line + len(pred.split("\n")) + 1  # + 1 for checking syntax errors
        context = context.replace("<PLACEHOLDER>", clean_lsp(pred, strip=True))
        uuid = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))
        tmp_py = f"{file_path[:-3]}_{uuid}.py"
        tmp_json = f"lint_{uuid}.json"
        try:
            with Path(tmp_py).open("w") as f:
                f.write(context)
            command = ["pylint", "--disable=C,R,W", f"--output-format=json:{tmp_json}", tmp_py]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            lint_json = []
            if len(stderr.decode().strip()) > 0:
                logging.error(f"error when pylint\n{stderr.decode()}")
            else:
                with Path(tmp_json).open("r") as f:
                    json_str = f.read().strip()
                if len(json_str) > 0:
                    lint_json = json.loads(json_str)
        finally:
            if Path(tmp_py).exists():
                os.unlink(tmp_py)
            if Path(tmp_json).exists():
                os.unlink(tmp_json)
        for err_item in lint_json:
            if err_item["line"] >= line and err_item["line"] <= end_line:
                errors.append((err_item['message-id'], err_item['symbol'], err_item['message']))
        lint_results.append(errors)
    return lint_results

def compute_metrics(tokenizer, inputs, contexts, expectations, predictions, lint_result_file=None, record_file=None):
    cleaned_preds = [clean_str(clean_lsp(code, strip=True)) for code in predictions]
    cleaned_expts = [clean_str(clean_lsp(expt, strip=False)) for expt in expectations]
    infos = []
    exact_match = 0
    for (docstr, signature), expt, pred, cleaned_expt, cleaned_pred in zip(inputs, expectations, predictions, cleaned_expts, cleaned_preds):
        info = f"======================\n[PROMPT]:\n{docstr}\n[Signature]:\n{signature}\n\n[EXPECTATION]:\n{expt}\n\n{cleaned_expt}\n\n[PREDICTION]:\n{pred}\n\n{cleaned_pred}\n\n"
        infos.append(info)
        if cleaned_expt == cleaned_pred:
            exact_match += 1
    exact_match /= len(inputs)
    logging.info(f"Exact Match: {exact_match}")

    
    if record_file is not None:
        with Path(f"{record_file}.basic").open("w") as f:
            f.write("\n\n".join(infos))

    logging.info("Calculating codebleu")
    codebleu = CodeBleu.compute_codebleu(cleaned_expts, cleaned_preds)

    logging.info("Calculating bleu-4")
    tokenized_preds = [tokenizer.tokenize(code) for code in cleaned_preds]
    tokenized_expts = [tokenizer.tokenize(code) for code in cleaned_expts]
    bleu = Bleu.compute_bleu(tokenized_expts, tokenized_preds, smooth=True)
    logging.info(f"[BLEU] blue-4: {round(bleu, 4)}, codebleu: {round(codebleu, 4)}") 

    logging.info("Calculating edit similarity")
    edit_sim = 0.0
    for pred, expt in zip(cleaned_preds, cleaned_expts):
        edit_sim += fuzz.ratio(" ".join(pred.split()), " ".join(expt.split()))
    edit_sim /= len(inputs) 
    logging.info(f"[EDIT] edit similarity: {round(edit_sim, 4)}")

    total_count, hit_count = 0, 0
    hit = []
    has_dep = []
    idx = 0
    for (docstr, signature), expt, pred  in tqdm(zip(inputs, expectations, predictions), desc="LSP Hit", total=len(inputs), ascii=True):
        cleaned_expt, cleaned_pred = clean_str(clean_lsp(expt)), clean_str(clean_lsp(pred))
        key_eles = set()
        lines = []
        for mobj in re.finditer(r"\W(\w+\.)*%s(\w+)\s*([^\w\.=]|$)" % re.escape(PLM_LSP_POINT), clean_str(expt)):
            key_ele = f"{mobj.group(1).strip() if mobj.group(1) else ''}{mobj.group(2).strip()}"
            # print(key_ele)

            if len(key_ele) == 0:
                continue
            obj_ref = key_ele.split(".")[0]
            if re.search(r"\s+%s(\W+\w+)*\s*=" % re.escape(obj_ref), cleaned_expt):  # local variable
                continue
            if re.search(r"as\s+%s\s*:" % re.escape(obj_ref), cleaned_expt):  # local variable
                continue
            if re.search(r"for\s+%s(\W+\w+)*\s+in" % re.escape(obj_ref), cleaned_expt): # local variable
                continue
            if re.search(r"from\s+%s(\W+\w+)*\s+import" % re.escape(key_ele), cleaned_expt): # local variable
                continue
            if "." not in key_ele and re.search(r"(\W|^)%s(\W|$)" % re.escape(obj_ref), signature): # function parameter
                continue
            key_eles.add(key_ele)
        _hit_count = 0
        for key_ele in key_eles:
            if re.search(r"(\W|^)%s(\W|$)" % re.escape(key_ele), cleaned_pred):
                _hit_count += 1
                lines.append(f"(√) {key_ele}\n")
            else:
                lines.append(f"(x) {key_ele}\n")
        total_count += len(key_eles)
        hit_count += _hit_count
        if len(key_eles) > 0:
            hit.append(_hit_count / len(key_eles))
        has_dep.append(True if len(key_eles) > 0 else False)

        infos[idx] = f"{infos[idx]}[LSP HIT]\n{''.join(lines)}\n"
        idx += 1
        if record_file is not None and idx % 50 == 0:
            with Path(f"{record_file}.lsphit").open("w") as f:
                f.write("\n\n".join(infos))
    if record_file is not None:
        with Path(f"{record_file}.lsphit").open("w") as f:
            f.write("\n\n".join(infos))
    hit_rate = hit_count / total_count if total_count > 0 else 0
    logging.info(f"[Macro LSP HIT] total: {total_count}, hit: {hit_count}, rate: {round(hit_rate, 4)}")
    logging.info(f"[Micro LSP HIT] total: {len(hit)}, rate: {round(sum(hit) / len(hit), 4)}")

    
    if lint_result_file is None or not Path(lint_result_file).exists():
        lint_results = lint(contexts, predictions)
        if lint_result_file is not None:
            with Path(lint_result_file).open("w") as f:
                json.dump(lint_results, f, indent=4)
    else:
        with Path(lint_result_file).open("r") as f:
            lint_results = json.load(f)

    syntax_pass, semantic_pass = 0, 0
    dep_syntax_pass, dep_semantic_pass = 0, 0
    for idx, errors in tqdm(enumerate(lint_results), desc="Lint", total=len(inputs), ascii=True):
        lines = []
        err_symbols = set()
        for err_id, err_symbol, err_msg in errors:
            err_symbols.add(err_symbol)
            lines.append(f"{err_id} {err_symbol}: {err_msg}\n")
        if "syntax-error" not in err_symbols:
            syntax_pass += 1
            if has_dep[idx]:
                dep_syntax_pass += 1
            if "no-member" not in err_symbols and "undefined-variable" not in err_symbols:
                semantic_pass += 1
                if has_dep[idx]:
                    dep_semantic_pass += 1
        info = "PASS!" if len(lines) == 0 else "".join(lines)
        infos[idx] = f"{infos[idx]}[LINT]\n{info}\n\n"
        idx += 1
        if record_file is not None and idx % 50 == 0:
            with Path(f"{record_file}").open("w") as f:
                f.write("\n\n".join(infos))
    if record_file is not None:
        with Path(f"{record_file}").open("w") as f:
            f.write("\n\n".join(infos))
    syntax_pass_rate = syntax_pass / len(inputs)
    semantic_pass_rate = semantic_pass / len(inputs)
    logging.info(f"[LINT] total: {len(inputs)}, syntax pass: {syntax_pass}, syntax pass rate: {syntax_pass_rate}")
    logging.info(f"[LINT] total: {len(inputs)}, semantic pass: {semantic_pass}, semantic pass rate: {semantic_pass_rate}")
    
    ## compute for the functions that have dependencies
    dep_total = len([d for d in has_dep if d])
    dep_syntax_pass_rate = dep_syntax_pass / dep_total
    dep_semantic_pass_rate = dep_semantic_pass / dep_total
    logging.info(f"[LINT-DEP] total: {dep_total}, syntax pass: {dep_syntax_pass}, syntax pass rate: {dep_syntax_pass_rate}")
    logging.info(f"[LINT-DEP] total: {dep_total}, semantic pass: {dep_semantic_pass}, semantic pass rate: {dep_semantic_pass_rate}")
    
    os.unlink(f"{record_file}.basic")
    os.unlink(f"{record_file}.lsphit")
    return exact_match, bleu, codebleu, edit_sim, hit_rate, syntax_pass_rate, semantic_pass_rate