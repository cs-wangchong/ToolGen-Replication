#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
from pathlib import Path
import json
import math
import random
import re
import traceback

from tqdm import tqdm
from multiprocessing import Pool

from coder.data_augmentor import DataAugmentor
from coder.utils.log_utils import init_log



if __name__ == '__main__':
    WORKER = 48

    REPO_DIR = "CodeSearchNet/resources/data/python/final/repos"
    OUTPUT_DIR = "data/augmentation"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    REGEX = re.compile(r"%s/.*?#.*?/" % re.escape(REPO_DIR))

    all_processed = set()
    for batch_file in Path(OUTPUT_DIR).glob("batch-*.jsonl"):
        processed = set()
        with batch_file.open("r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                example = json.loads(line)
                processed.add(REGEX.search(example["file_path"]).group(0)[:-1])
        batch_id = re.search(r"batch-(\d*)\.jsonl", str(batch_file)).group(1)
        processed_path = Path(f"{OUTPUT_DIR}/processed-{batch_id}.json")
        if processed_path.exists():
            with processed_path.open("r") as f:
                processed.update(json.load(f))
        with processed_path.open("w") as f:
            json.dump(list(processed), f, indent=4)
        all_processed.update(processed)


    projects = [str(path) for path in Path(REPO_DIR).glob("*")]
    projects = [path for path in projects if path not in all_processed]
    random.shuffle(projects)

    print(len(all_processed), len(projects))

    def error_callback(error):
        print(error, flush=True)

    def handle(worker_id, batch):
        processed_path = Path(f"{OUTPUT_DIR}/processed-{worker_id}.json")
        processed = set()
        if processed_path.exists():
            with processed_path.open("r") as f:
                processed.update(json.load(f))
        
        for pj_path in tqdm(batch, ascii=True, desc=f"worker-{worker_id}: "):
            processed.add(pj_path)
            with processed_path.open("w") as f:
                json.dump(list(processed), f, indent=4)
            # logging.info(f"worker-{worker_id}: {pj_path}")
            try:
                examples = DataAugmentor.handle_project(pj_path)
            except Exception as e:
                continue
            if len(examples) == 0:
                continue
            json_lines = [json.dumps(example) for example in examples]
            with Path(f"{OUTPUT_DIR}/batch-{worker_id}.jsonl").open("a") as f:
                f.write("\n".join(json_lines) + "\n")
                f.flush()
            
        return processed

    batch_size = math.ceil(len(projects) / WORKER)
    pool = Pool(WORKER)
    procs = []
    for batch_id in range(WORKER):
        batch = projects[batch_id * batch_size: batch_id * batch_size + batch_size]
        proc = pool.apply_async(handle, (batch_id + 1, batch), error_callback=error_callback)
        procs.append(proc)

    pool.close()
    pool.join()