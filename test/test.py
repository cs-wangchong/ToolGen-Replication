import json
import re

code = '''def wrap_into_list(x):
            return [<lsp>x]
        else:
            return list(<lsp>x)'''

def clean_lsp(code:str):
    code = re.sub(r" ?%s ?" % re.escape("<lsp>"), "", code)
    return code

print(clean_lsp(code))