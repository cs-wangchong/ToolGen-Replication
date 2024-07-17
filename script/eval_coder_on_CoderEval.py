
from collections import defaultdict
import datetime
import logging
from pathlib import Path
import os
import sys
import json
import re
import argparse
import subprocess
import numpy as np
import multiprocessing
import time

from tqdm import tqdm

from coder.utils.log_utils import init_log
from coder.model import *
from coder.generator import Generator
from coder.constants import *



class Process(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as exception:
            self._cconn.send(exception)

    def join(self, timeout):
        super().join(timeout)

        if self.is_alive():
            self.terminate()
        super().join()

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def clean_docstring(docstring):
    # docstring = docstring.strip("\"'").strip()
    docstring = docstring.split("\n\n")[0].strip()
    docstring = docstring.split(":param")[0].strip()
    docstring = docstring.split(":return")[0].strip()
    docstring = re.sub(r"\s*\n\s*[A-Z].*$", "", docstring, re.S)
    docstring = re.sub(r"\s+", " ", docstring)
    docstring = docstring.split(". ")[0]
    return docstring

def clean_lsp(code:str):
    regexp = r"(\w) %s ?" % re.escape(PLM_LSP_POINT)
    code = re.sub(regexp, r"\1 ", code)
    regexp = r" %s ?" % re.escape(PLM_LSP_POINT)
    code = re.sub(regexp, "", code)
    regexp = r"%s" % re.escape(PLM_LSP_POINT)
    code = re.sub(regexp, "", code)
    return code

def clean_lsp(code:str, strip=True):
    if strip:
        regexp = r" %s ?" % re.escape(PLM_LSP_POINT)
    else:
        regexp = r"%s" % re.escape(PLM_LSP_POINT)
    code = re.sub(regexp, "", code)
    return code

def clean_code(code:str):
    return re.sub(r"'''(.*)'''", "", code)


MODEL_FACTORY = {
    "codegen": ("Salesforce/codegen-350M-mono", init_codegen, 2048),
    "deepseek": ("deepseek-ai/deepseek-coder-1.3b-base", init_deepseek, 16384),
    "codellama": ("codellama/CodeLlama-7b-Python-hf", init_codellama, 4096),
    "deepseek-lora": ("deepseek-ai/deepseek-coder-1.3b-base", init_deepseek_lora, 16384),
    "codellama-lora": ("codellama/CodeLlama-7b-Python-hf", init_codellama_lora, 4096),
    "codet5": ("Salesforce/codet5p-220m-py", init_codet5, 512),
    "codegpt": ("microsoft/CodeGPT-small-py", init_codegpt, 1024),
}


CoderEval_ROOT = "CoderEval"
N = 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("-model", "--model", required=True, type=str)
    parser.add_argument("-dir", "--dir", required=False, type=str, default=None)
    # parser.add_argument("-stamp", "--timestamp", required=False, type=str, default=None)
    # parser.add_argument("-lsp", "--lsp", required=False, default=False, action="store_true")
    # parser.add_argument("-sample", "--sample", required=False, default=False, action="store_true")
    
    parser.add_argument("-lsp_conf", "--lsp_conf", required=False, type=float, default=0.5)
    parser.add_argument("-token_conf", "--token_conf", required=False, type=float, default=0.0)
    parser.add_argument("-token_k", "--token_k", required=False, type=float, default=None)
    parser.add_argument("-temp", "--temp", required=False, type=float, default=1.0)
    parser.add_argument("-max_len", "--max_len", required=False, type=int, default=192)
    parser.add_argument("-batch", "--batch", required=False, type=int, default=24)
    args = parser.parse_args()

    
    DIR = args.dir
    # SAMPLE = args.sample
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

    init_log(f"{DIR}/testing-CoderEval/testing.log", logging.INFO)
    PRED_RESULT_FILE = f"{DIR}/testing-CoderEval/predictions.jsonl"
    # LINT_RESULT_FILE = f"{DIR}/testing-CoderEval/lintresults.json"
    # RECORD_FILE = f"{DIR}/testing-CoderEval/record.txt"

    train_repos = set()
    for jsonl in Path("CodeSearchNet/resources/data/python/final/jsonl/train").rglob("*.jsonl"):
        for line in jsonl.open("r"):
            d = json.loads(line.strip())
            train_repos.add(d["repo"])
        
    with Path(f"{CoderEval_ROOT}/CoderEval4Python.json").open("r") as f:
        samples = json.load(f)["RECORDS"]

    test_examples = []
    for sample in samples:
        if sample['project'] in train_repos:
            continue
        repo = f"{CoderEval_ROOT}/repos/{sample['project']}" 
        file_path = f"{repo}/{sample['file_path']}"
        
        beg_lineno, end_lineno = int(sample['lineno']), int(sample['end_lineno'])

        content_lines = sample['file_content'].split("\n")
        indent = re.match(r"\s*", content_lines[beg_lineno-1]).group(0)
        code_lines = [content_lines[beg_lineno-1][len(indent):]] + content_lines[beg_lineno:end_lineno]
        code = "\n".join(code_lines)
        context_lines = content_lines[:beg_lineno-1] + [indent + '<PLACEHOLDER>'] + content_lines[end_lineno:]
        context = "\n".join(context_lines)

        line = beg_lineno
        column = len(indent)

        docstr = clean_docstring(sample['docstring'])
        signature = re.search(r"def\s+(%s\s*\(.*?\)(\s*->.*?)?:)" % re.escape(sample['name']), code, re.M|re.S)
        if signature is None:
            continue
        signature = signature.group(1)
        idx = code.index(signature) + len(signature)
        prefix = code[:idx]
        body = "\n".join(line for line in code[idx:].split("\n") if line.strip() != "")
        indent = re.match(r"\s*", body.split("\n")[0]).group(0)
        prefix = f"{prefix}\n{indent}'''{docstr}'''\n"
        signature = signature.strip(" :")
        test_examples.append((sample['_id'], repo, file_path, context, line, column, docstr, signature, code, prefix, body))

    logging.info(json.dumps(test_examples, indent=4))
    
    
    logging.info(f"model: {MODEL_NAME}")
    logging.info(f"checkpoint: {CHECKPOINT}")
    logging.info(f"dataset: CoderEval")
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

    print(type(model))

    generator = Generator(model, tokenizer, MODEL_MAX_LEN)

    predictions = []

    total_time = 0
    total_lsp = 0

    if not CALL_LSP:
        batch_ranges = list(zip(range(0, len(test_examples), BATCH_SIZE), range(BATCH_SIZE, len(test_examples)+BATCH_SIZE, BATCH_SIZE)))
        for beg, end in tqdm(batch_ranges, ascii=True, desc="Evaluation"):
            _ids = []
            batch = []
            for _id, repo, file_path, context, line, column, docstr, signature, code, prefix, body in test_examples[beg:end]:
                _ids.append(_id)
                batch.append({
                    "path": file_path,
                    "context": context,
                    "line": line,
                    "column": column,
                    "docstr": docstr,
                    "signature": signature,
                    "prefix": prefix,
                })
            stime = time.time()
            outputs = generator.generate_simple(batch, max_len=MAX_LEN, repetition_penalty=REPETITION_PENALTY)
            total_time += time.time() - stime
            predictions.extend([(_id, clean_code(output)) for _id, output in zip(_ids, outputs)])
    else:
        repo2insts = defaultdict(list)
        _ids = []
        for idx, (_id, repo, file_path, context, line, column, docstr, signature, code, prefix, body) in enumerate(test_examples, 1):
            _ids.append(_id)
            repo2insts[repo].append((idx, file_path, context, line, column, docstr, signature, code, prefix, body))
        for repo, insts in tqdm(repo2insts.items(), total=len(repo2insts), ascii=True, desc="Evaluation"):
            generator.update_lsp_project(repo)
            batch_ranges = list(zip(range(0, len(insts), BATCH_SIZE), range(BATCH_SIZE, len(insts)+BATCH_SIZE, BATCH_SIZE)))
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
                stime = time.time()
                outputs = generator.generate_with_lsp(
                    batch,
                    max_len=MAX_LEN,
                    lsp_conf=LSP_CONF,
                    repetition_penalty=REPETITION_PENALTY
                )
                total_time += time.time() - stime
                predictions.extend(zip(idxes, outputs))
                for output in outputs:
                    total_lsp += output.count(PLM_LSP_POINT)
        predictions = [clean_code(clean_lsp(pred)) for _, pred in sorted(predictions, key=lambda p: p[0])]
        predictions = [(_id, pred) for _id, pred in zip(_ids, predictions)]
    
    lines = [json.dumps({"_id": _id, "generate_results": [pred]}) for _id, pred in predictions]
    with Path(PRED_RESULT_FILE).open("w") as f:
        f.write("\n".join(lines))

    
    ### copied from CoderEval, did some modifications
    count_tot = len(test_examples)
    dict_std_nonestd={f"{CoderEval_ROOT}/repos/standalone/neo4j-_meta-deprecated.py":f"{CoderEval_ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_meta_deprecated_passk_validte.py",
                  f"{CoderEval_ROOT}/repos/standalone/neo4j-work-query-unit_of_work.py":f"{CoderEval_ROOT}/repos/neo4j---neo4j-python-driver/src/neo4j/_work/query_unit_of_work_passk_validte.py",
                  f"{CoderEval_ROOT}/repos/standalone/krake-krake-controller-kubernetes-hooks-on.py":f"{CoderEval_ROOT}/repos/rak-n-rok---Krake/krake/krake/controller/kubernetes/hooks_on_passk_validte.py"}

    
    fw = open(PRED_RESULT_FILE + "_out.jsonl", 'w')
    
    listtot = []
    collection = {sample["_id"]: sample for sample in samples}

    kk = 0
    project_path = f"{CoderEval_ROOT}/repos/"
    dict_id_file={}
    generate_list = []
    for keyy in collection:
        dictTemp = collection[keyy]
        save_data = project_path + "standalone/" + dictTemp["file_path"].replace(".py", "").replace("/", "-") + "-" + \
                    dictTemp["name"] + ".py"
        if save_data in dict_std_nonestd.keys():
            save_data = dict_std_nonestd[save_data]
            if Path(save_data).exists():
                kk += 1
                dict_id_file[dictTemp["_id"]] = save_data
        elif Path(save_data).exists():
            kk+=1
            dict_id_file[dictTemp["_id"]] = save_data
        else:
            file_path = dictTemp['file_path']
            if project_path + dictTemp["project"].replace("/", "---") == f"{CoderEval_ROOT}/repos/neo4j---neo4j-python-driver":
                save_data = os.path.join(project_path + dictTemp['project'].replace("/", "---") + "/src",
                                         file_path).replace(
                    ".py", "_" + dictTemp["name"] + "_passk_validte.py")
            else:
                save_data = os.path.join(project_path + dictTemp['project'].replace("/", "---"), file_path).replace(
                    ".py", "_" + dictTemp["name"] + "_passk_validte.py")
            if save_data in dict_std_nonestd.keys():
                save_data = dict_std_nonestd[save_data]

            if os.path.exists(save_data):
                kk+=1
                dict_id_file[dictTemp["_id"]] = save_data


    with open(PRED_RESULT_FILE, 'r') as fr:
        list_tot_question = fr.readlines()
    list_count_tot = []
    dict_level_tot = {}
    tot_k = []
    for i in range(0, N):
        tot_k.append(0.0)
    record_out = {}
    for i in range(0, len(list_tot_question)):
        dictTemp = {}
        ques = json.loads(list_tot_question[i])
        content_doc = collection[ques["_id"]]
        if content_doc is None:
            continue
        dictTemp["file_path"] = content_doc["file_path"]
        if "project" in content_doc.keys():
            dictTemp["project"] = content_doc["project"]
        dictTemp["name"] = content_doc["name"]
        dictTemp["docstring"] = content_doc["docstring"]
        dictTemp["_id"] = str(ques['_id'])
        solutions = ques["generate_results"]
        list_code = []
        for solution in solutions:
            list_code.append(solution)
        dictTemp['code'] = list_code
        level = content_doc["level"]
        dictTemp["level"] = level
        if level not in dict_level_tot.keys():
            dict_level_tot[level] = 1
        else:
            dict_level_tot[level] += 1
        generate_list.append(dictTemp)
        f_save_data = open(dict_id_file[str(ques['_id'])], 'r')
        file_content = f_save_data.read()
        f_save_data.close()
        file_content_list = file_content.split("\n")
        import ast
        tka=0
        ast_file = ast.parse(file_content)
        start_indent = 0
        new_data = ""
        for node in ast.walk(ast_file):
            if isinstance(node, ast.FunctionDef):
                temp_method_name = node.name
                if content_doc["name"] != temp_method_name and "_"+content_doc["name"]!=temp_method_name:
                    continue
                start_line = node.lineno
                end_line = node.end_lineno
                indent_s = file_content_list[start_line - 1]
                tttt = indent_s.lstrip(" ")
                start_indent = len(indent_s) - len(tttt)
                new_data=""
                for i in range(0, start_line - 1):
                    new_data += file_content_list[i]
                    new_data += "\n"
                new_data += "<insert generated code here>\n"
                for i in range(end_line, len(file_content_list)):
                    new_data += file_content_list[i]
                    new_data += "\n"
        assert new_data!=""
        list_generate_code = []
        c = 0
        code_num = 0
        for code in list_code:
            dict_temp = {}
            dict_temp["generate_code"] = code
            code_list = code.split("\n")
            tttt = code_list[0].lstrip(" ")
            code_indent = len(code_list[0]) - len(tttt)
            new_code = ""
            if start_indent > code_indent:
                str_a = ""
                for iii in range(0, start_indent - code_indent):
                    str_a += " "
                for ccc in code_list:
                    ttz = str_a + ccc
                    new_code += ttz
                    new_code += "\n"
            else:
                new_code = code
            out_data = new_data.replace("<insert generated code here>", new_code)
            save_data_new=dict_id_file[str(ques['_id'])]
            f = open(save_data_new.replace(".py", str(code_num) + ".py"), 'w')
            f.write(out_data)
            f.close()
            try:
                process = subprocess.Popen([sys.executable, save_data_new.replace(".py", str(code_num) + ".py")],
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output, error = process.communicate(timeout=30)
            except:
                code_num += 1
                continue
            if process.returncode == 0:
                dict_temp["is_pass"] = True
                c += 1
            else:
                dict_temp["is_pass"] = False
            dict_temp["return_code"] = process.returncode
            code_num += 1
            list_generate_code.append(dict_temp)

        if level not in record_out.keys():
            temp_tot_k = []
            for tti in range(0, N):
                temp_tot_k.append(0.0)
        else:
            temp_tot_k = record_out[level]
        dictTemp["generate_results"] = list_generate_code
        fw.write(json.dumps(dictTemp) + "\n")
        fw.flush()
        for k in range(1, N + 1):
            if N - c < k:
                tot_k[k - 1] += 1.0
                temp_tot_k[k - 1] += 1.0
            else:
                tot_k[k - 1] += (1.0 - np.prod(1.0 - k / np.arange(N - c + 1, N + 1)))
                temp_tot_k[k - 1] += (1.0 - np.prod(1.0 - k / np.arange(N - c + 1, N + 1)))
            logging.info(f'{dictTemp["_id"]} {N} {c} {tot_k[k - 1]}')
        record_out[level] = temp_tot_k
    fw.close()

    logging.info("\n")
    logging.info("\n")
    logging.info(f'## total: {count_tot}')
    for k, tt in enumerate(tot_k, 1):
        logging.info(f"pass@{k}: {round(tt / count_tot * 100, 1)}% ({int(tt)})")
    logging.info("\n")

    for key in ["self_contained", "slib_runnable", "plib_runnable", "class_runnable", "file_runnable", "project_runnable"]:
        tot_k = record_out[key]
        logging.info(f'## {key}: {dict_level_tot[key]}')
        for k, tt in enumerate(tot_k, 1):
            logging.info(f"pass@{k}: {round(tt / dict_level_tot[key] * 100, 1)}% ({int(tt)})")
        logging.info("\n")

    
    logging.info(f"AVERAGE TIME: {total_time / len(test_examples)}")
    logging.info(f"AVERAGE LSP: {total_lsp / len(test_examples)}")
