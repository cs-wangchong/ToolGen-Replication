#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path
import json
import re
import pickle
from transformers import RobertaTokenizer
from tqdm import tqdm
import random
from multiprocessing import Pool
import ast

from coder.data_augmentor import DataAugmentor
from coder.constants import *

def clean_docstring(docstring):
    docstring = docstring.strip()[3:-3]
    docstring = docstring.strip("\"'").strip()
    docstring = docstring.split("\n\n")[0].strip()
    docstring = docstring.split(":param")[0].strip()
    docstring = docstring.split(":return")[0].strip()
    docstring = re.sub(r"\s*\n\s*[A-Z].*$", "", docstring, re.S)
    docstring = re.sub(r"\s+", " ", docstring)
    docstring = docstring.split(". ")[0]
    return docstring

def validate_docstring(docstring):
    if len(docstring) == 0:
        return False
    lower_docstring = docstring.lower()
    if lower_docstring[0] not in set("abcdefghijklmnopqrstuvwxyz"):
        return False
    words = lower_docstring.split(" ")
    if not (3 <= len(words) <= 256):
        return False
    if any(len(word) > 25 for word in words):
        return False
    if lower_docstring.startswith("deprecated"):
        return False
    if lower_docstring.startswith("todo"):
        return False
    if lower_docstring.startswith("note"):
        return False
    if lower_docstring.startswith("test"):
        return False
    if lower_docstring.startswith("overrides"):
        return False
    if lower_docstring.startswith("we "):
        return False
    if lower_docstring.startswith("you "):
        return False
    if lower_docstring.startswith("this "):
        return False
    if lower_docstring.endswith("?"):
        return False
    if lower_docstring.endswith("!"):
        return False
    if re.match(r"\w+:", lower_docstring):
        return False
    if "<img " in lower_docstring or "https://" in lower_docstring:
        return False
    if "---" in lower_docstring or "=" in lower_docstring:
        return False
    if " -- " in lower_docstring or "=" in lower_docstring:
        return False
    return True

def clean_code(code):
    # code = re.sub(r"'(.*?)'", STR, code)
    # code = re.sub(r'"(.*?)"', STR, code)
    return code.strip()

def validate_code(code):
    if code.startswith("def test_"):
        return False
    if len(code.split("\n")) == 1:
        return False
    # lines = code.split("\n")
    # last_line = lines[-1].strip()
    # if last_line.startswith("raise NotImplementedError"):
    #     return False
    # if len(lines) == 2 and last_line == "pass":
    #     return False
    return True

def validate_body(body:str):
    body = body.strip()
    if body == "raise NotImplementedError":
        return False
    if body == "raise NotImplementedError()":
        return False
    if body == "pass":
        return False
    if body == "return":
        return False
    return True

def get_lsp_context(file_path: str, signature:str):
    try:
        with Path(file_path).open("r") as f:
            code = f.read()
        tree = ast.parse(code)
    except Exception as e:
        return None, None, None
    funcs = (func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef))
    
    for func in funcs:
        func_source = ast.get_source_segment(code, func)
        line, column = func.lineno, func.col_offset
        context = code.replace(func_source, "<PLACEHOLDER>")
        _, tokens = DataAugmentor.tokenize_func(func_source)
        original_func = "".join(tokens).replace(PLM_LSP_POINT, "<unk>").strip()

        if re.search(r"def\s+%s" % re.escape(signature), original_func):
            return context, line, column
    return None, None, None

def handle(jsonl_path:str, max_len=128, tokenizer_name="Salesforce/codet5p-220m-py"):
    REPO_PATTERN = re.compile(r"/([^/]+#[^/]+)/")

    train_examples, valid_examples, test_examples = list(), list(), list()
    augmented_train_examples, augmented_valid_examples, augmented_test_examples = list(), list(), list()
    lsp_test_examples = list()
    kept_count = 0
    too_long_count = 0
    
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    counter = 0
    for line in tqdm(Path(jsonl_path).open("r").readlines(), ascii=True):
        d = json.loads(line.strip())
        repo = REPO_PATTERN.search(d["file_path"]).group(1).replace("#", "/")

        docstring = clean_docstring(d["docstring"])
        if not validate_docstring(docstring):
            continue

        code = d["code"].replace(PLM_LSP_POINT, "<unk>")
        code = clean_code(code)
        if not validate_code(code):
            continue
        code: str

        signature = re.search(r"def\s+(%s\s*\(.*?\)(\s*->.*?)?:)" % re.escape(d['function_name']), code, re.M|re.S)
        if signature is None:
            continue
        signature = signature.group(1)

        context, line, column = get_lsp_context(d["file_path"], signature)
        if context is None:
            continue
        
        idx = code.index(signature) + len(signature)
        prefix = code[:idx]
        body = "\n".join(line for line in code[idx:].split("\n") if line.strip() != "")
        indent = re.match(r"\s*", body.split("\n")[0]).group(0)
        prefix = f"{prefix}\n{indent}'''{docstring}'''\n"

        if not validate_body(body):
            continue

        augmented_code = d["augmented_code"]
        augmented_code = augmented_code.replace(PLM_LSP_POINT, "<unk>")
        augmented_signature = re.search(r"def\s+((%s)\s*\((.*?)\)(\s*->.*?)?:)" % re.escape(d['function_name']), augmented_code, re.M|re.S)
        if augmented_signature is None:
            continue
        augmented_signature = augmented_signature.group(1)
        augmented_code = augmented_code.replace(augmented_signature, signature)
        if signature not in augmented_code:
            continue

        code_len = len(tokenizer.tokenize(code))
        if code_len > max_len:
            too_long_count += 1
            continue
        kept_count += 1
        
        doc_len = len(tokenizer.tokenize(docstring))
        
        lsp_dict = d["lsp_dict"]
        skipped_lsp_points = set()
        for lsp_point, completions in lsp_dict.items():
            completions = [c for c in completions if c not in SKIPPED]
            if len(completions) == 0:
                skipped_lsp_points.add(lsp_point)
            elif not re.search(r"(%s)(%s)(\W|$)" % (re.escape(lsp_point), "|".join(re.escape(c) for c in completions)), augmented_code):
                skipped_lsp_points.add(lsp_point)
        
        augmented_code = re.sub(r"%s" % "|".join(re.escape(p) for p in skipped_lsp_points), "", augmented_code)
        augmented_code = re.sub(r"<LSP-POINT-\d+>", PLM_LSP_POINT, augmented_code)
        augmented_code = clean_code(augmented_code)

        idx = augmented_code.index(signature) + len(signature)
        augmented_prefix = augmented_code[:idx]
        augmented_body = "\n".join(line for line in augmented_code[idx:].split("\n") if line.strip() != "")
        augmented_indent = re.match(r"\s*", augmented_body.split("\n")[0]).group(0)
        augmented_prefix = f"{augmented_prefix}\n{augmented_indent}'''{docstring}'''\n"

        lsp_count = augmented_code.count(PLM_LSP_POINT)

        signature = signature.strip(" :")

        if repo in train_repos:
            train_examples.append((docstring, signature, code, prefix, body, doc_len, code_len))
            augmented_train_examples.append((docstring, signature, augmented_code, augmented_prefix, augmented_body, lsp_count))
        elif repo in valid_repos:
            valid_examples.append((docstring, signature, code, prefix, body, doc_len, code_len))
            augmented_valid_examples.append((docstring, signature, augmented_code, augmented_prefix, augmented_body, lsp_count))
        elif repo in test_repos:
            test_examples.append((docstring, signature, code, prefix, body, doc_len, code_len))
            augmented_test_examples.append((docstring, signature, augmented_code, augmented_prefix, augmented_body, lsp_count))
            lsp_test_examples.append((repo, d["file_path"], context, line, column, docstring, signature, augmented_code, augmented_prefix, augmented_body))
        counter += 1
        # if counter > 20:
        #     break
    return (
        train_examples,
        valid_examples,
        test_examples,
        augmented_train_examples,
        augmented_valid_examples,
        augmented_test_examples,
        lsp_test_examples,
        kept_count,
        too_long_count,
    )

    


if __name__ == '__main__':
    train_repos, valid_repos, test_repos = set(), set(), set()
    for jsonl in Path("CodeSearchNet/resources/data/python/final/jsonl/train").rglob("*.jsonl"):
        for line in jsonl.open("r"):
            d = json.loads(line.strip())
            train_repos.add(d["repo"])
    for jsonl in Path("CodeSearchNet/resources/data/python/final/jsonl/valid").rglob("*.jsonl"):
        for line in jsonl.open("r"):
            d = json.loads(line.strip())
            valid_repos.add(d["repo"])
    for jsonl in Path("CodeSearchNet/resources/data/python/final/jsonl/test").rglob("*.jsonl"):
        for line in jsonl.open("r"):
            d = json.loads(line.strip())
            test_repos.add(d["repo"])
    with Path("data/datasets/train_repos.json").open("w") as f:
        json.dump(list(train_repos), f, indent=4)
    with Path("data/datasets/valid_repos.json").open("w") as f:
        json.dump(list(valid_repos), f, indent=4)
    with Path("data/datasets/test_repos.json").open("w") as f:
        json.dump(list(test_repos), f, indent=4)


    with Path("data/datasets/train_repos.json").open("r") as f:
        train_repos = set(json.load(f))
    with Path("data/datasets/valid_repos.json").open("r") as f:
        valid_repos = set(json.load(f))
    with Path("data/datasets/test_repos.json").open("r") as f:
        test_repos = set(json.load(f))

    jsonl_paths = [str(jsonl) for jsonl in Path("data/augmentation").rglob("batch-*.jsonl")]

    pool = Pool(len(jsonl_paths))
    procs = []
    for jsonl_path in jsonl_paths:
        proc = pool.apply_async(handle, (jsonl_path, 128, "Salesforce/codet5p-220m-py"),)
        procs.append(proc)

    pool.close()
    pool.join()

    train_examples, valid_examples, test_examples = list(), list(), list()
    augmented_train_examples, augmented_valid_examples, augmented_test_examples = list(), list(), list()
    lsp_test_examples = list()
    kept_count = 0
    too_long_count = 0

    for proc in procs:
        (
            _train_examples,
            _valid_examples,
            _test_examples,
            _augmented_train_examples,
            _augmented_valid_examples,
            _augmented_test_examples,
            _lsp_test_examples,
            _kept_count,
            _too_long_count,
        ) = proc.get()
        train_examples.extend(_train_examples)
        valid_examples.extend(_valid_examples)
        test_examples.extend(_test_examples)
        augmented_train_examples.extend(_augmented_train_examples)
        augmented_valid_examples.extend(_augmented_valid_examples)
        augmented_test_examples.extend(_augmented_test_examples)
        lsp_test_examples.extend(_lsp_test_examples)
        kept_count += _kept_count
        too_long_count += _too_long_count
    
    train_tuples = [(docstring, signature, code, prefix, body, augmented_code, augmented_prefix, augmented_body, doc_len, code_len, lsp_count) for (docstring, signature, code, prefix, body, doc_len, code_len), (_, _, augmented_code, augmented_prefix, augmented_body, lsp_count) in zip(train_examples, augmented_train_examples)]
    valid_tuples = [(docstring, signature, code, prefix, body, augmented_code, augmented_prefix, augmented_body, doc_len, code_len, lsp_count) for (docstring, signature, code, prefix, body, doc_len, code_len), (_, _, augmented_code, augmented_prefix, augmented_body, lsp_count) in zip(valid_examples, augmented_valid_examples)]
    test_tuples = [(docstring, signature, code, prefix, body, augmented_code, augmented_prefix, augmented_body, doc_len, code_len, lsp_count) for (docstring, signature, code, prefix, body, doc_len, code_len), (_, _, augmented_code, augmented_prefix, augmented_body, lsp_count) in zip(test_examples, augmented_test_examples)]
    random.shuffle(train_tuples)

    train_examples, augmented_train_examples = [], []
    visited_train_prompts = set()
    train_doc_len = 0
    train_code_len = 0
    train_lsp_count = 0
    for docstring, signature, code, prefix, body, augmented_code, augmented_prefix, augmented_body, doc_len, code_len, lsp_count in train_tuples:
        if (docstring, signature) in visited_train_prompts:
            continue
        visited_train_prompts.add((docstring, signature))
        train_doc_len += doc_len
        train_code_len += code_len
        train_lsp_count += lsp_count

        train_examples.append((docstring, signature, code, prefix, body))
        augmented_train_examples.append((docstring, signature, augmented_code, augmented_prefix, augmented_body))


    valid_examples, augmented_valid_examples = [], []
    valid_doc_len = 0
    valid_code_len = 0
    valid_lsp_count = 0
    for docstring, signature, code, prefix, body, augmented_code, augmented_prefix, augmented_body, doc_len, code_len, lsp_count in valid_tuples:
        if (docstring, signature) in visited_train_prompts:
            continue
        valid_doc_len += doc_len
        valid_code_len += code_len
        valid_lsp_count += lsp_count
        valid_examples.append((docstring, signature, code, prefix, body))
        augmented_valid_examples.append((docstring, signature, augmented_code, augmented_prefix, augmented_body))

    test_examples, augmented_test_examples = [], []
    test_doc_len = 0
    test_code_len = 0
    test_lsp_count = 0
    for docstring, signature, code, prefix, body, augmented_code, augmented_prefix, augmented_body, doc_len, code_len, lsp_count in test_tuples:
        if (docstring, signature) in visited_train_prompts:
            continue
        test_doc_len += doc_len
        test_code_len += code_len
        test_lsp_count += lsp_count
        test_examples.append((docstring, signature, code, prefix, body))
        augmented_test_examples.append((docstring, signature, augmented_code, augmented_prefix, augmented_body))

    lsp_test_examples = [(repo, path, context, line, column, docstring, signature, code, prefix, body) for repo, path, context, line, column, docstring, signature, code, prefix, body in lsp_test_examples if (docstring, signature) not in visited_train_prompts]

    with Path("data/datasets/meta_info.txt").open("w") as f:
        f.write(f"dataset sizes: {len(train_examples)}, {len(valid_examples)}, {len(test_examples)}\n")
        f.write(f"augmented dataset sizes: {len(augmented_train_examples)}, {len(augmented_valid_examples)}, {len(augmented_test_examples)}\n")
        f.write(f"long function ratio: {too_long_count / (kept_count + too_long_count)}\n")
        f.write("\n\n")
        f.write(f"===== Training set =====\n")
        f.write(f"average docstring length: {train_doc_len / len(train_examples)}\n")
        f.write(f"average code length: {train_code_len / len(train_examples)}\n")
        f.write(f"average lsp count: {train_lsp_count / len(train_examples)}\n\n")
        f.write(f"===== Validataion set =====\n")
        f.write(f"average docstring length: {valid_doc_len / len(valid_examples)}\n")
        f.write(f"average code length: {valid_code_len / len(valid_examples)}\n")
        f.write(f"average lsp count: {valid_lsp_count / len(valid_examples)}\n\n")
        f.write(f"===== Testing set =====\n")
        f.write(f"average docstring length: {test_doc_len / len(test_examples)}\n")
        f.write(f"average code length: {test_code_len / len(test_examples)}\n")
        f.write(f"average lsp count: {test_lsp_count / len(test_examples)}\n\n")

            # CODE_PATTERN = re.compile(r"(.*?)('''|\"\"\")", re.S)
            # mobj = CODE_PATTERN.match(d["code"])
            # if mobj is None:
            #     continue
            # signature = mobj.group(1).strip()
            # docstring = mobj.group(1).split("\n")[-1] + f'""" ' + d["docstring"].split("\n\n")[0].strip() + ' """'
            # augmented_body = "\n".join(d["augmented_code"].replace(signature, "").split("\n")[1:])
            # body = augmented_body.replace("<extra_id_99>", "")

            # if repo in train_repos:
            #     train_examples.add((docstring, signature, body))
            #     augmented_train_examples.add((docstring, signature, augmented_body))
            # elif repo in valid_repos:
            #     valid_examples.add((docstring, signature, body))
            #     augmented_valid_examples.add((docstring, signature, augmented_body))
            # elif repo in test_repos:
            #     test_examples.add((docstring, signature, body))
            #     augmented_test_examples.add((docstring, signature, augmented_body))

    with Path("data/datasets/train.bin").open("wb") as f:
        pickle.dump(list(train_examples), f)
    with Path("data/datasets/valid.bin").open("wb") as f:
        pickle.dump(list(valid_examples), f)
    with Path("data/datasets/test.bin").open("wb") as f:
        pickle.dump(list(test_examples), f)

    with Path("data/datasets/augmented-train.bin").open("wb") as f:
        pickle.dump(list(augmented_train_examples), f)
    with Path("data/datasets/augmented-valid.bin").open("wb") as f:
        pickle.dump(list(augmented_valid_examples), f)
    with Path("data/datasets/augmented-test.bin").open("wb") as f:
        pickle.dump(list(augmented_test_examples), f)
    
    with Path("data/datasets/train.json").open("w") as f:
        json.dump(list(train_examples), f, indent=4)
    with Path("data/datasets/valid.json").open("w") as f:
        json.dump(list(valid_examples), f, indent=4)
    with Path("data/datasets/test.json").open("w") as f:
        json.dump(list(test_examples), f, indent=4)

    with Path("data/datasets/augmented-train.json").open("w") as f:
        json.dump(list(augmented_train_examples), f, indent=4)
    with Path("data/datasets/augmented-valid.json").open("w") as f:
        json.dump(list(augmented_valid_examples), f, indent=4)
    with Path("data/datasets/augmented-test.json").open("w") as f:
        json.dump(list(augmented_test_examples), f, indent=4)


    train_examples = random.sample(train_examples, 50000)
    valid_examples = random.sample(valid_examples, 3000)

    augmented_train_examples = random.sample(augmented_train_examples, 50000)
    augmented_valid_examples = random.sample(augmented_valid_examples, 3000)

    with Path("data/datasets/train-sample50000.bin").open("wb") as f:
        pickle.dump(list(train_examples), f)
    with Path("data/datasets/valid-sample3000.bin").open("wb") as f:
        pickle.dump(list(valid_examples), f)

    with Path("data/datasets/augmented-train-sample50000.bin").open("wb") as f:
        pickle.dump(list(augmented_train_examples), f)
    with Path("data/datasets/augmented-valid-sample3000.bin").open("wb") as f:
        pickle.dump(list(augmented_valid_examples), f)
    
    with Path("data/datasets/train-sample50000.json").open("w") as f:
        json.dump(list(train_examples), f, indent=4)
    with Path("data/datasets/valid-sample3000.json").open("w") as f:
        json.dump(list(valid_examples), f, indent=4)

    with Path("data/datasets/augmented-train-sample50000.json").open("w") as f:
        json.dump(list(augmented_train_examples), f, indent=4)
    with Path("data/datasets/augmented-valid-sample3000.json").open("w") as f:
        json.dump(list(augmented_valid_examples), f, indent=4)


    with Path("data/datasets/lsp-test.bin").open("wb") as f:
        pickle.dump(list(lsp_test_examples), f)
    with Path("data/datasets/lsp-test.json").open("w") as f:
        json.dump(list(lsp_test_examples), f, indent=4)

    lsp_test_examples = random.sample(lsp_test_examples, 100)
    with Path("data/datasets/lsp-test-sample100.bin").open("wb") as f:
        pickle.dump(lsp_test_examples, f)
    with Path("data/datasets/lsp-test-sample100.json").open("w") as f:
        json.dump(list(lsp_test_examples), f, indent=4)


    with Path("data/datasets/lsp-test.bin").open("rb") as f:
        lsp_test_examples = pickle.load(f)


    lsp_test_examples = random.sample(lsp_test_examples, 100)
    with Path("data/datasets/lsp-test-sample100.bin").open("wb") as f:
        pickle.dump(lsp_test_examples, f)
    with Path("data/datasets/lsp-test-sample100.json").open("w") as f:
        json.dump(list(lsp_test_examples), f, indent=4)