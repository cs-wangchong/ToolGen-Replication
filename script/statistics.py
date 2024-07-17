from pathlib import Path
import pickle
import json

all_repos = set()
for batch_file in Path("data/augmentation").glob("processed-*.jsonl"):
    with batch_file.open("r") as f:
        for item in json.load(f):
            all_repos.add(item.split("/")[-1].replace("#", "/"))
        
train_repos = set(json.load(Path("data/datasets/train_repos.json").open("r")))

print(len(train_repos - all_repos))