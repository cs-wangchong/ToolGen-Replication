from pathlib import Path
import json
import os

with Path('data/datasets/test_repos.json').open("r") as f:
    repos = json.load(f)

DIR = "/home/wangchong/Workspace/Coder-LSP/CodeSearchNet/resources/data/python/final/repos"

for repo in repos:
    os.system(f"cp -r {DIR}/{repo.replace('/', '#')} /home/wangchong/Workspace/Coder-LSP/data/test_repos")

