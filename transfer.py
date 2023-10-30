from pathlib import Path


existing_repos = set()
for repo in Path("CodeSearchNet/resources/data/python/final/repos").glob("*"):
    existing_repos.add(repo.parts[-1])



with open("CodeSearchNet/resources/data/python/final/repo_list.txt", "r") as f:
    all_repos = list({line.strip().split(" ")[-1] for line in f.readlines()})
    all_repos.sort()

repos = []
for repo in all_repos:
    if repo in existing_repos:
        continue
    repos.append(repo)

print(len(repos))

cmd = f"scp -r {' '.join(repos)}  wangchong_cuda12@10.96.183.249:/home/wangchong_cuda12/Workspace/Coder-LSP/CodeSearchNet/resources/data/python/final/repos"

Path("scp_cmd.sh").open("w").write("#!/bin/zsh\n" + cmd)