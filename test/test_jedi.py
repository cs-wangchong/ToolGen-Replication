import jedi


pj_path = "/home/ubuntu/Workspace/Coder-LSP/test_projects/you-get-b746ac01c9f39de94cac2d56f665285b0523b974"
env_path = "/home/ubuntu/anaconda3/envs/coder-lsp-testenv/bin/python"

project = jedi.Project(pj_path, environment_path=env_path, added_sys_path=(pj_path,))
# project.save()
# project.load(basepath)

code = "from ..common import *\n"
cur_file = pj_path + "/" + "src/you_get/extractors/youtube.py"


script = jedi.Script(code=code, path=cur_file, project=project)

completions = script.complete(line=2, column=0)
print(len(completions))
for completion in completions:
    print(completion.complete)


    