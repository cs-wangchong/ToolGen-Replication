import logging
import re
from typing import Callable, Union
import warnings
import copy
import inspect

import jedi
import torch

from peft import PeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer, AddedToken, RobertaTokenizer
from transformers import PreTrainedModel, AutoModelForCausalLM, LlamaForCausalLM, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, CausalLMOutputWithPast
from transformers.generation.configuration_utils import GenerationConfig

from coder.utils.trie import Trie
from coder.constants import *

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
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, build_prompt:Callable):
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.build_prompt = build_prompt
        self.jedi_pj = None
        self.model.eval()
        self.device = model.device
    
    def generate_simple(
            self,
            docstrs,
            signatures,
            max_len=512,
            repetition_penalty=1.0
        ):
        prompts = [self.build_prompt(docstr, signature) for docstr, signature in zip(docstrs, signatures)]
        input_ids = self.tokenizer(prompts, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        if isinstance(self.model, PeftModelForCausalLM):
            input_ids = input_ids[:,:-1]   # important: remove </s>
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_length=max_len+input_ids.size(1),
            repetition_penalty=repetition_penalty,
        )
        outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
        if isinstance(self.model, (PeftModelForCausalLM, GPT2LMHeadModel)):
            outputs = [output[len(prompt):].strip() for output, prompt in zip(outputs, prompts)]
        outputs = [clean_pad(output) for output in outputs]
        return outputs
    
    @torch.no_grad()
    def generate_with_lsp(
            self,
            docstrs,
            signatures,
            file_paths,
            code_contexts,
            lines,
            columns,
            max_len=256,
            lsp_conf=0.8,
            beam_size=1,
            repetition_penalty=1.0,
            verbose=True,
        ):

        # prepare inputs
        prompts = [self.build_prompt(docstr, signature) for docstr, signature in zip(docstrs, signatures)]
        input_ids = self.tokenizer(prompts, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        if isinstance(self.model, PeftModelForCausalLM):
            input_ids = input_ids[:,:-1]   # important: remove </s>
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        model_kwargs = {
            "attention_mask": attention_mask
        }
        
        if self.model.generation_config._from_model_config and self.model.generation_config._original_object_hash == hash(
            self.model.generation_config
        ):
            new_generation_config = GenerationConfig.from_model_config(self.model.config)
            if new_generation_config != self.model.generation_config:
                warnings.warn(
                    "You have modified the pretrained model configuration to control generation. This is a"
                    " deprecated strategy to control generation and will be removed soon, in a future version."
                    " Please use and modify the model generation configuration (see"
                    " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                )
                self.model.generation_config = new_generation_config
            generation_config = self.model.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**model_kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(model_kwargs.copy())

        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            input_ids, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        if not self.model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if self.model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        if self.model.config.is_encoder_decoder:
            input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        

        output_ids = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size, dtype=torch.long, device=self.device)
        best_probs, best_scores = torch.zeros((batch_size,), device=self.device), torch.zeros((batch_size,), device=self.device)

        nodes = [None] * batch_size
        funcs = [""] * batch_size
        lsps = [(None, None, None)] * batch_size
        lsp_logs = [[] for _ in range(batch_size)]

        finished = set()
        for _ in range(max_len + input_ids.size(1)):
            for idx in range(batch_size):
                if idx in finished:
                    continue
                func = self.tokenizer.decode(output_ids[idx], skip_special_tokens=True).strip()
                funcs[idx] = func

                if func.endswith(PLM_LSP_POINT) and best_probs[idx].item() >= lsp_conf:
                    cleaned_func = clean_lsp(func, strip=True)
                    file_path, code_context, line, column = file_paths[idx], code_contexts[idx], lines[idx], columns[idx]
                    cands = self.get_lsp_completions(cleaned_func, file_path, code_context, line, column)
                    if len(cands) == 0:
                        continue
                    lsps[idx] = (func, best_probs[idx].item(), cands)
                    trie = Trie()
                    for cand in cands:
                        trie.insert([self._token2id(t) for t in self.tokenizer.tokenize(cand)])
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
                    next_ids.append([self.tokenizer.pad_token_id])
                    continue
                _probs = probs[idx]
                if funcs[idx].endswith(PLM_LSP_POINT):
                    _probs[self._token2id(PLM_LSP_POINT)] = 0.
                if nodes[idx] is not None and nodes[idx].is_valid and not nodes[idx].is_end_of_sequence:
                    children = nodes[idx].get_children()
                    children = [(child, _probs[child.key].item()) for child in children]
                    children.sort(key=lambda p: p[1], reverse=True)
                    next_id = children[0][0].key
                    nodes[idx] = children[0][0]
                    lsp_func, lsp_prob, lsp_cands = lsps[idx]
                    lsp_logs[idx].append((lsp_func, lsp_prob, lsp_cands, children))
                else:
                    next_id = torch.argmax(_probs, dim=-1, keepdim=False).item()
                    nodes[idx] = None
                next_ids.append([next_id])
                best_probs[idx] = _probs[next_id]
                best_scores[idx] += torch.log(_probs[next_id])
                if next_id in {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}:
                    finished.add(idx)
            
            next_ids = torch.tensor(next_ids, dtype=torch.long, device=self.device)     
            input_ids = torch.cat([input_ids, next_ids], -1)
            output_ids = torch.cat([output_ids, next_ids], -1)
            
            if len(finished) == batch_size:
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
        completions = [
            completion for completion in completions if len(completion) > 0 and completion not in SKIPPED
        ]
        return completions