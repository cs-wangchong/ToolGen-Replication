import os

if __name__ == "__main__":
    DIR = "/home/wangchong/Workspace/CodeSearchNet/resources/data/python/final"

    with open(f"{DIR}/repos.txt", "r") as f:
        repos = [tuple(repo.strip().split("\t")) for repo in f]

    for repo, sha in repos:
        os.system(f"git clone --depth 1 git@github.com:{repo} {DIR}/repos/{repo.replace('/', '#')}")