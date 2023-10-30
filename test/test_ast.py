import ast
import re

if __name__ == "__main__":

    path = "/home/wangchong/Workspace/Coder-LSP/test_projects/you-get-b746ac01c9f39de94cac2d56f665285b0523b974/src/you_get/extractors/bigthink.py"
    code = open(path, "r").read()
    lines = code.split()

    tree = ast.parse(code)
    functions = (func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef))
    pairs = []
    for func in functions:
        source = ast.get_source_segment(code, func)
        docstring = ast.get_docstring(func, clean=False)
        if docstring is None:
            continue
        regex = r"\n(\t| )*('''%s'''|\"\"\"%s\"\"\")\s*?\n" % (re.escape(docstring), re.escape(docstring))
        cleaned_source = re.sub(regex, "\n", source, re.S)
        docstring = docstring.strip()
        pairs.append((cleaned_source, docstring))

    print(pairs)
    
    # language = Language("coder/tree_sitter.so", "python")
    # parser = Parser()
    # parser.set_language(language)


    # tree = parser.parse(code.encode())
    # query = language.query("""
    # (function_definition) @function.def
    # """)

    # captures = query.captures(tree.root_node)
    # for capture, _  in captures:
    #     print(code.encode()[capture.start_byte:capture.end_byte].decode())