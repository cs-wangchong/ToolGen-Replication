import logging
import re
from typing import Callable, List, Union
import warnings
import copy
import inspect

import jedi
import torch

from peft import PeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer, AddedToken, RobertaTokenizer
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers import LlamaForCausalLM, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, CausalLMOutputWithPast
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import is_torchdynamo_compiling

from coder.utils.trie import Trie
from coder.constants import *
from coder.prompt import *

torch.manual_seed(42)  # pytorch random seed

def clean_pad(code:str):
    code = re.sub(r" ?%s ?" % re.escape("<pad>"), "", code)
    return code

def clean_lsp(code:str, strip=True):
    if strip:
        regexp = r" %s ?" % re.escape(PLM_LSP_POINT)
    else:
        regexp = r"%s" % re.escape(PLM_LSP_POINT)
    code = re.sub(regexp, "", code)
    return code


def clean_str(code):
    code = re.sub(r"'(.*?)'", "", code)
    code = re.sub(r'"(.*?)"', "", code)
    return code.strip()


class Generator:
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, model_max_length=1024):
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.jedi_pj = None
        self.model.eval()
        self.model_max_length = min(model_max_length, self.tokenizer.model_max_length)
        self.device = model.device
        self.all_special_ids = set(self.tokenizer.all_special_ids)
        self.lsp_id = self.tokenizer.convert_tokens_to_ids(PLM_LSP_POINT)
    
    def generate_simple(
            self,
            inputs: List[dict],
            max_len=512,
            repetition_penalty=1.0 
        ):
        if self.model.config.is_encoder_decoder:
            prompts = [build_prompt_encoder_decoder(inp["docstr"], inp["signature"]) for inp in inputs]
        else:
            prompts = [build_prompt_decoder_only(inp["prefix"]) for inp in inputs]
        input_ids = self.tokenizer(prompts, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        if input_ids.shape[1] >= self.model_max_length:
            return [f"def {inst['signature']}:pass" for inst in inputs]
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_length=min(self.model_max_length, input_ids.shape[1] + max_len),
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
        # if not self.model.config.is_encoder_decoder:
        #     outputs = [output[len(prompt):].strip() for output, prompt in zip(outputs, prompts)]
        outputs = [clean_pad(output) for output in outputs]
        return outputs
        

    @torch.no_grad()
    def generate_with_lsp(
            self,
            inputs,
            # file_paths,
            # code_contexts,
            # lines,
            # columns,
            max_len=256,
            lsp_conf=0.8,
            beam_size=1,
            repetition_penalty=1.0,
            rm_lsp=True,
            verbose=True,
        ):
        if self.model.config.is_encoder_decoder:
            prompts = [build_prompt_encoder_decoder(inp["docstr"], inp["signature"]) for inp in inputs]
        else:
            prompts = [build_prompt_decoder_only(inp["prefix"]) for inp in inputs]
        file_paths = [inp["path"] for inp in inputs]
        code_contexts = [inp["context"] for inp in inputs]
        lines = [inp["line"] for inp in inputs]
        columns = [inp["column"] for inp in inputs]
        # prepare inputs
        input_ids = self.tokenizer(prompts, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        if input_ids.shape[1] >= self.model_max_length:
            return [f"def {inst['signature']}:pass" for inst in inputs]
        
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        model_kwargs = {"attention_mask": attention_mask}

        input_ids, model_kwargs = self._prepare(input_ids, model_kwargs)
        batch_size = input_ids.shape[0]

        # output_ids = [[] for _ in range(batch_size)]
        if self.model.config.is_encoder_decoder:
            output_ids = [[] for _ in range(batch_size)]
        else:
            output_ids = [self.tokenizer.encode(prompt, add_special_tokens=False, padding=False, truncation=False) for prompt in prompts]
        best_probs, best_scores = torch.zeros((batch_size,), device=self.device), torch.zeros((batch_size,), device=self.device)

        nodes = [None] * batch_size
        accumulated_infos = [(0, 0)] * batch_size
        funcs = []
        for idx in range(batch_size):
            func = self.tokenizer.decode(output_ids[idx], skip_special_tokens=False).strip()
            if func.startswith(self.tokenizer.bos_token):
                func = func[len(self.tokenizer.bos_token):].strip()
            funcs.append(func)
        lsps = [(None, None, None)] * batch_size
        lsp_logs = [[] for _ in range(batch_size)]

        cand_cache = dict()

        finished = set()
        for _ in range(max_len):

            for idx in range(batch_size):
                if idx in finished:
                    continue

                # row = input_ids[idx]
                # print(row[-1].item() == self.lsp_id)
                # if rm_lsp and row[-1].item() == self.lsp_id:
                #     new_row = torch.zeros_like(row)
                #     new_row[0] = self.tokenizer.pad_token_id
                #     new_row[1:] = row[:-1]
                #     input_ids[idx] = new_row
                func = funcs[idx]
                if func.endswith(PLM_LSP_POINT) and best_probs[idx].item() >= lsp_conf:
                    cleaned_func = clean_lsp(func, strip=True)
                    file_path, code_context, line, column = file_paths[idx], code_contexts[idx], lines[idx], columns[idx]
                    cands = self.get_lsp_completions(cleaned_func, file_path, code_context, line, column)
                    if len(cands) == 0:
                        continue
                    lsps[idx] = (func, best_probs[idx].item(), cands)
                    trie = Trie()
                    # speed up
                    func_len = None
                    for cand in cands:
                        if cand not in cand_cache:
                            if func_len is None:
                                func_len = len(self.tokenizer.tokenize(func))
                            # tokenization results may be different when using candidates solely or using candidates within func
                            tokens = self.tokenizer.tokenize(func + cand)[func_len:]
                            cand_cache[cand] = [self._token2id(t) for t in tokens]
                        trie.insert(cand_cache[cand])
                        
                    nodes[idx] = trie.root
            

            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.model(**model_inputs)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            
            logits = outputs.logits[:,-1,:]
            probs = torch.softmax(logits, -1)

            next_ids = []
            for idx in range(batch_size):
                if idx in finished:
                    next_ids.append(self.tokenizer.pad_token_id)
                    continue
                _probs = probs[idx]
                if funcs[idx].endswith(PLM_LSP_POINT):
                    _probs[self._token2id(PLM_LSP_POINT)] = 0.
                # no lsp tokens involved
                if nodes[idx] is None or len(nodes[idx].get_children()) == 0:
                    next_id = torch.argmax(_probs, dim=-1, keepdim=False).item()
                    nodes[idx] = None
                    accumulated_infos[idx] = (0, 0)
                else:
                    children = nodes[idx].get_children()
                    children = [(child, _probs[child.key].item()) for child in children]
                    children.sort(key=lambda p: p[1], reverse=True)
                    best_child, score = children[0]
                    acc_score, acc_len = accumulated_infos[idx]
                    # the current node is not only an end_of_sequence but also an internal node, but treat it as an internal node
                    if nodes[idx].is_end_of_sequence and acc_score / acc_len > (acc_score + score) / (acc_len + 1):
                        next_id = torch.argmax(_probs, dim=-1, keepdim=False).item()
                        nodes[idx] = None
                        accumulated_infos[idx] = (0, 0)
                        lsp_logs[idx].append((lsp_func, lsp_prob, lsp_cands, children))
                        # if :
                        #     next_id = best_child.key
                        #     nodes[idx] = best_child
                        #     lsp_func, lsp_prob, lsp_cands = lsps[idx]
                        #     accumulated_infos[idx] = (acc_score + score, acc_len + 1)
                        #     lsp_logs[idx].append((lsp_func, lsp_prob, lsp_cands, children))
                        # else:  # treat the current node as the end of the sequence                                                        
                            
                    else: # the current node is absolutely an internal node
                        next_id = best_child.key
                        nodes[idx] = best_child
                        lsp_func, lsp_prob, lsp_cands = lsps[idx]
                        accumulated_infos[idx] = (acc_score + score, acc_len + 1)
                        lsp_logs[idx].append((lsp_func, lsp_prob, lsp_cands, children))
                    
                next_ids.append(next_id)
                best_probs[idx] = _probs[next_id]
                best_scores[idx] += torch.log(_probs[next_id])
                if next_id in {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}:
                    finished.add(idx)
               
            input_ids = torch.cat([input_ids, torch.tensor([[_id] for _id in next_ids], dtype=torch.long, device=self.device)], -1)

            for beam_ids, next_id in zip(output_ids, next_ids):
                if next_id in self.all_special_ids:
                    continue
                beam_ids.append(next_id)
            
            for idx in range(batch_size):
                func = self.tokenizer.decode(output_ids[idx], skip_special_tokens=False).strip()
                if func.startswith(self.tokenizer.bos_token):
                    func = func[len(self.tokenizer.bos_token):].strip()
                funcs[idx] = func
            
            if len(finished) == batch_size or input_ids.shape[1] >= self.model_max_length:
                break
        if verbose:
            for idx in range(batch_size):
                if len(lsp_logs[idx]) == 0:
                    continue
                logging.info(f"================================================")
                last_func = None
                level = 1
                for (func, prob, cands, children) in lsp_logs[idx]:
                    if func != last_func:
                        logging.info(f"[ENTER LSP] {prob} >= {lsp_conf}")
                        logging.info(f"\tuncomplete func: ```{repr(func)}```")
                        logging.info(f"\tlsp candidates: {cands}")
                        last_func = func
                        level = 1
                    for child, _prob in children[:5]:
                        logging.info('\t' * level + f"\ttoken: {self._id2token(child.key)}, prob: {_prob}")
                    level += 1
        funcs = [clean_pad(func) for func in funcs]
        return funcs


    
    def _prepare(self, inputs, kwargs):
        '''copied from transformers.generation.utils.GenerationMixin.generate()'''
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self.model._validate_model_class()
        generation_config, model_kwargs = self.model._prepare_generation_config(None, **kwargs)
        self.model._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self.model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.model.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logging.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if self.model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.model.config.is_encoder_decoder:
            input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        return input_ids, model_kwargs
    

    def _id2token(self, id):
        return self.tokenizer.convert_ids_to_tokens([id])[0]

    def _token2id(self, token):
        return self.tokenizer.convert_tokens_to_ids([token])[0]
    
    def update_lsp_project(self, pj_path, py_env=None):
        if self.jedi_pj is None or self.jedi_pj._path != pj_path:
            self.jedi_pj: jedi.Project = jedi.Project(pj_path, environment_path=py_env, added_sys_path=(pj_path,))
    
    def get_lsp_completions(self, func, file_path, code_context, line, column):
        lines = func.split("\n")
        _line, _column = line, column
        if len(lines) == 1:
            _column += len(lines[0])
        else:
            _line += len(lines) - 1
            _column = len(lines[-1])
        context = code_context.replace("<PLACEHOLDER>", func)
        # logging.info(f"context: ```\n{context}\n```")
        try:
            script = jedi.Script(project=self.jedi_pj, path=file_path, code=context)
            completions = script.complete(line=_line, column=_column)
        except Exception as e:
            completions = []
        completions = [completion.complete.strip() for completion in completions]
        # completions = [
        #     completion for completion in completions if len(completion) > 0 and completion not in SKIPPED
        # ]
        completions = [
            completion for completion in completions if len(completion) > 0
        ]
        if all(comp in SKIPPED for comp in completions):
            return []
        return completions