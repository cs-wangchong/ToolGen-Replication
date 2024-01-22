#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path
import ast
import logging
import re
from io import StringIO
import  tokenize

import jedi
from coder.constants import SKIPPED


LSP_POINT = "<LSP-POINT-%d>"

class DataAugmentor:

    @staticmethod
    def handle_project(project_path, py_env=None, max_file=500):
        
        examples = []
        py_files = list(Path(project_path).rglob("*.py"))
        if len(py_files) > max_file:
            return examples
        
        project: jedi.Project = jedi.Project(project_path, environment_path=py_env, added_sys_path=(project_path,))
        for path in py_files:
            # if path.parts[-1] != "bigthink.py":
            #     continue
            logging.info(f"file: {str(path)}")
            for name, func_source, augmented_func_source, lsp_dict, docstr in DataAugmentor.handle_file(project, str(path)):
                examples.append({
                    "file_path": str(path),
                    "function_name": name,
                    "code": func_source,
                    "augmented_code": augmented_func_source,
                    "lsp_dict": lsp_dict,
                    "docstring": docstr
                })
        return examples

    @staticmethod
    def handle_file(project: jedi.Project, file_path: str, file_max_line=1000, func_max_line=50, func_max_token=512,):
        with Path(file_path).open("r") as f:
            code = f.read()
        if len(code.split("\n")) > file_max_line:
            return
        try:
            tree = ast.parse(code)
        except Exception as e:
            return
        funcs = (func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef))
        
        for func in funcs:
            func_source = ast.get_source_segment(code, func)
            line, column = func.lineno, func.col_offset

            rest_source = code.replace(func_source, "<PLACEHOLDER>")
            docstring, tokens = DataAugmentor.tokenize_func(func_source)
            if docstring.strip().strip('\'"') == "":
                continue
            original_func = "".join(tokens)
            if len(original_func.split("\n")) > func_max_line or len(tokens) > func_max_token:
                continue 
            # logging.info("##########################")
            # logging.info(docstring)
            # logging.info(original_func)
            augmented_func, lsp_dict = DataAugmentor.augment_func(project, file_path, tokens, rest_source, line, column)
            yield (func.name, original_func, augmented_func, lsp_dict, docstring)


    @staticmethod
    def augment_func(
            project: jedi.Project,
            file_path: str,
            tokens: list,
            rest_source: str,
            line: int,
            column: int
        ):
         
        augmented_func = ""
        incomplete_func = ""
        lsp_dict = dict()
        lsp_idx = 1
        for token in tokens:
            if token == "\n":
                line += 1
                column = 0
            elif token.strip() == "" or token in SKIPPED:
                column += len(token)
            else:
                incomplete_code = rest_source.replace("<PLACEHOLDER>", incomplete_func)
                try:
                    script = jedi.Script(project=project, path=file_path, code=incomplete_code)
                    completions = script.complete(line=line, column=column)
                    completions = [completion.complete.strip() for completion in completions]
                    completions = [completion for completion in completions if len(completion) > 0 and completion not in SKIPPED]
                except Exception as e:
                    completions = []
                # logging.info("######")
                # logging.info(f"token: {token}")
                # logging.info(f"incomplete func: {incomplete_func}")
                # logging.info(f"completions: {completions}")
                if token in set(completions):
                    trigger_token = LSP_POINT % lsp_idx
                    augmented_func += trigger_token
                    lsp_dict[trigger_token] = completions
                    lsp_idx += 1

                column += len(token)
            incomplete_func += token
            augmented_func += token
        return augmented_func, lsp_dict
            
            

    @staticmethod
    def tokenize_func(func_source):
        io_obj = StringIO(func_source)
        tokens = []
        doc = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                tokens.append(" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            tokens.append(token_string)
                        else:
                            doc += token_string
                    else:
                        doc += token_string
                else:
                    doc += token_string
            else:
                tokens.append(token_string)
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        cleaned_tokens = []
        counter = 0
        for t in tokens:
            if t == "\n":
                if counter == 0:
                    pass
                elif "".join(cleaned_tokens[-counter:]).strip() == "":
                    cleaned_tokens = cleaned_tokens[:-counter]
                else:
                    cleaned_tokens.append(t)
                counter = 0
            else:
                cleaned_tokens.append(t)
                counter += 1
        return doc, cleaned_tokens

    
    # def handle_file(self, project: jedi.Project, file_path: str, max_line=1000):
    #     with Path(file_path).open("r") as f:
    #         code = f.read()
    #     if len(code.split("\n")) > max_line:
    #         return
    #     try:
    #         tree = ast.parse(code)
    #     except Exception as e:
    #         return
    #     funcs = (func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef))
        
    #     for func in funcs:
            # docstr = ast.get_docstring(func, clean=False)
            # logging.info(f"func name: {func.name}")
            # # logging.info(f"docstring:\n{docstr}")
            # if docstr is None or len(docstr.strip()) == 0:
            #     continue
            # func_source, augmented_func_source = self.augment_func(project, file_path, code, func, docstr)
            # if func_source is None:
            #     continue
            # yield (func.name, func_source, augmented_func_source, docstr.strip())

    # def augment_func(
    #     self,
    #     project: jedi.Project,
    #     file_path: str,
    #     code: str,
    #     func: ast.FunctionDef,
    #     docstr: str,
    #     max_line: int = 50,
    #     max_len: int = 512,
        
    # ):
    #     left, right, body_source = DataAugmentor.parse_func_body(code, func)
    #     cleaned_body_source = DataAugmentor.remove_docstr(body_source, docstr)
    #     if len(cleaned_body_source.split("\n")) > max_line:
    #         return None, None
    #     # logging.info(f"body_source:\n{body_source}")
    #     # logging.info(f"cleaned_body_source:\n{cleaned_body_source}")

    #     line_no, col_no = func.body[0].lineno, 0

    #     body_tokens = self.tokenizer.tokenize(cleaned_body_source)
    #     if len(body_tokens) > max_len:
    #         return None, None
    #     # logging.info(f"body_tokens: {body_tokens}")

    #     idx = 0

    #     def __update_left(span):
    #         nonlocal left, line_no, col_no
    #         left += span
    #         lines = span.split("\n")
    #         line_no +=  len(lines) - 1
    #         if len(lines) == 1:
    #             col_no += len(span)
    #         else:
    #             col_no = len(lines[-1])

    #     def __skip_spaces(offset=0):
    #         nonlocal idx
    #         while idx + offset < len(body_tokens):
    #             next_token = self.tokenizer.convert_tokens_to_string([body_tokens[idx + offset]])
                
    #             if next_token.strip() == "":
    #                 __update_left(next_token)
    #                 idx += 1
    #             else:
    #                 break

    #     while idx + 1 < len(body_tokens):  
            
    #         __skip_spaces(offset=0)
        
    #         token = body_tokens[idx]
    #         token_str = self.tokenizer.convert_tokens_to_string([token])
    #         __update_left(token_str)

    #         __skip_spaces(offset=1)

    #         # logging.info(f"incomplete code:\n" + left + " \n" + right)
    #         # logging.info(f"position: {(line_no, col_no)}")

    #         if idx + 1 >= len(body_tokens):
    #             break
            
    #         try:
    #             script = jedi.Script(project=project, path=file_path, code=left + "\n" + right)
    #             completions = script.complete(line=line_no, column=col_no)
                
    #             completions = [completion.complete.strip() for completion in completions]
    #             completions = [completion for completion in completions if len(completion) > 0]
    #             if len(completions) == 0:
    #                 script = jedi.Script(project=project, path=file_path, code=left + " \n" + right)
    #                 completions = script.complete(line=line_no, column=col_no + 1)
    #                 completions = [completion.complete.strip() for completion in completions]
    #                 completions = [completion for completion in completions if len(completion) > 0]
    #         except Exception as e:
    #             completions = []

    #         # logging.info(f"completions: {completions}")
    #         for completion in completions:
    #             completion_len = len(self.tokenizer.tokenize(completion))
    #             next_span = self.tokenizer.convert_tokens_to_string(body_tokens[idx+1: idx+1+completion_len])
    #             if next_span.strip() == completion.strip():
    #                 body_tokens.insert(idx+1, self.mark_token)
    #                 idx += 1 + completion_len
    #                 __update_left(next_span)
    #                 break
    #         idx += 1
    #     func_source = ast.get_source_segment(code, func)
    #     augmented_body_source = self.tokenizer.convert_tokens_to_string(body_tokens)
    #     augmented_func_source = func_source.replace(body_source, augmented_body_source)
    #     return func_source, augmented_func_source


    # @staticmethod
    # def remove_docstr(source, docstr):
    #     regex = r"(\t| )*('''%s'''|\"\"\"%s\"\"\")\s*?\n" % (re.escape(docstr), re.escape(docstr))
    #     source = re.sub(regex, "", source, re.S)
    #     return source

    # @staticmethod
    # def parse_func_body(code: str, func: ast.FunctionDef) -> str:
    #     lineno = func.body[0].lineno - 1
    #     end_lineno = func.body[-1].end_lineno - 1
    #     col_offset = func.body[0].col_offset
    #     end_col_offset = func.body[-1].end_col_offset

    #     lines = ast._splitlines_no_ff(code)
    #     left = ''.join(lines[:lineno])
    #     right = ''.join(lines[end_lineno+1:])

    #     # logging.info(f"left:\n{left}")
    #     # logging.info(f"right:\n{right}")

    #     if end_lineno == lineno:
    #         return left, right, lines[lineno].encode()[col_offset:end_col_offset].decode()

    #     first = lines[lineno]
    #     last = lines[end_lineno].encode()[:end_col_offset].decode()
    #     lines = lines[lineno+1:end_lineno]

    #     lines.insert(0, first)
    #     lines.append(last)
    #     return left, right, ''.join(lines)






    

    