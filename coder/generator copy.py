import json
import logging
import random
from pathlib import Path
import re
import os
import subprocess
from typing import Union, Callable
import string

import ast
import jedi

from tqdm import tqdm
import torch
from peft import PeftModelForCausalLM
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from transformers import AutoTokenizer, PreTrainedTokenizer, AddedToken, RobertaTokenizer
from transformers import PreTrainedModel, AutoModelForCausalLM, LlamaForCausalLM, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, CausalLMOutputWithPast

from coder.data_augmentor import DataAugmentor
from coder.utils.metric_utils import Bleu, CodeBleu
from coder.utils.trie import Trie
from coder.constants import *

torch.manual_seed(42)  # pytorch random seed

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
        if isinstance(self.model, (PeftModelForCausalLM, GPT2LMHeadModel)):
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

        return outputs

    def generate_with_lsp(
            self,
            docstr,
            signature,
            file_path,
            code_context,
            line,
            column,
            max_len=256,
            lsp_threshold=0.8,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6,
            repetition_penalty=1.0,
        ):
        prompt = self.build_prompt(docstr, signature)
        input_ids = self.tokenizer([prompt], add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        logits_processor = LogitsProcessorList()
        if repetition_penalty > 1:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        encoder_outputs, decoding_ids, past_key_values, prob, score = self._init_decoding(input_ids, attention_mask)

        for _ in range(1, max_len + 1):
            # logging.info(self.tokenizer.decode(decoding_ids[0], skip_special_tokens=False).strip())
            if sum(decoding_ids[0].eq(self.tokenizer.eos_token_id).long()) >= 1: # type: ignore
                break
            lsp_called = False
            func = self.tokenizer.decode(decoding_ids[0], skip_special_tokens=True)
            # logging.info(func)
            if func.strip().endswith(PLM_LSP_POINT) and prob >= lsp_threshold:
                cleaned_func = clean_lsp(func, strip=True)
                cands = self.get_lsp_completions(cleaned_func, file_path, code_context, line, column)
                logging.info(f"[ENTER LSP] confidence: {prob} >= {lsp_threshold}")
                logging.info(f"\tuncomplete func: ```{repr(func)}```")
                logging.info(f"\tcurrent score: {score}")
                logging.info(f"\tlsp candidates: {cands}")

                if len(cands) > 0:
                    decoding_ids, past_key_values, prob, _score = self._lsp_step(
                        cands,  
                        decoding_ids,
                        attention_mask,
                        encoder_outputs,
                        past_key_values,
                        logits_processor,
                        token_threshold=token_threshold,
                        token_k=token_k,
                        temperature=temperature
                    )
                    if decoding_ids is not None:
                        score += _score
                        lsp_called = True
            if not lsp_called:
                outputs = self._model_forward(decoding_ids, attention_mask, past_key_values, encoder_outputs)
                logits = outputs.logits[:,-1,:]
                probs = torch.softmax(logits, -1).view(-1)
                # logging.info(probs[self._token2id(PLM_LSP_POINT)].item())
                if func.strip().endswith(PLM_LSP_POINT):
                    probs[self._token2id(PLM_LSP_POINT)] = 0.
                best_idx = torch.argmax(probs, dim=-1, keepdim=False)
                decoding_ids = torch.cat([decoding_ids, best_idx.unsqueeze(0).unsqueeze(0)], -1)
                past_key_values = outputs.past_key_values
                best_prob = probs[best_idx]
                best_score = torch.log(best_prob)
                prob = best_prob.item()
                score += best_score.item()

        func = self.tokenizer.decode(decoding_ids[0], skip_special_tokens=True).strip()
        logging.info(f"[PREDICTION] score: {score}")
        logging.info(f"\n{func}")
        logging.info("")
        return func
    
    def _id2token(self, id):
        return self.tokenizer.convert_ids_to_tokens([id])[0]

    def _token2id(self, token):
        return self.tokenizer.convert_tokens_to_ids([token])[0]

    def _init_decoding(self, input_ids, attention_mask):
        # logging.info(input_ids)
        if isinstance(self.model, (PeftModelForCausalLM, GPT2LMHeadModel)):
            decoding_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device)
            outputs: CausalLMOutputWithPast = self.model(
                input_ids = input_ids[:,:-1],           # important: remove </s>
                attention_mask = attention_mask[:,:-1], # important: remove </s>
                return_dict = True
            )
            encoder_outputs = None
        elif isinstance(self.model, T5ForConditionalGeneration):
            decoding_ids = torch.tensor([[self.model.generation_config.decoder_start_token_id]], dtype=torch.long, device=self.device)
            outputs: Seq2SeqLMOutput = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                decoder_input_ids = decoding_ids,
                output_hidden_states = True,
                return_dict = True
            )
            encoder_outputs = BaseModelOutput(
                last_hidden_state=outputs.encoder_last_hidden_state,
                hidden_states=None,
                attentions=None,
            )
        logits = outputs.logits[:,-1,:]
        probs = torch.softmax(logits, -1).view(-1)
        best_idx = torch.argmax(probs, dim=-1, keepdim=False)
        decoding_ids = torch.cat([decoding_ids, best_idx.unsqueeze(0).unsqueeze(0)], -1)
        past_key_values = outputs.past_key_values
        best_prob = probs[best_idx]
        best_score = torch.log(best_prob)
        return encoder_outputs, decoding_ids, past_key_values, best_prob.item(), best_score.item()
    
    def _model_forward(self, decoding_ids, attention_mask, past_key_values, encoder_outputs=None) -> Union[CausalLMOutputWithPast,Seq2SeqLMOutput]:
        # logging.info(decoding_ids)
        if isinstance(self.model, (PeftModelForCausalLM, GPT2LMHeadModel)):
            outputs: CausalLMOutputWithPast = self.model(
                input_ids = decoding_ids[:,-1:],
                past_key_values = past_key_values,
                return_dict = True
            )
        elif isinstance(self.model, T5ForConditionalGeneration):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state.repeat(decoding_ids.size(0), 1, 1),
                hidden_states=None,
                attentions=None,
            )
            outputs: Seq2SeqLMOutput = self.model(
                encoder_outputs = encoder_outputs,
                attention_mask = attention_mask,
                decoder_input_ids = decoding_ids[:,-1:],
                past_key_values = past_key_values,
                return_dict = True
            )
        return outputs

    def _lsp_step(
            self,
            all_cands,
            decoding_ids,
            attention_mask,
            encoder_outputs,
            past_key_values,
            logits_processor,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6
        ):

        if token_k:
            token_k = min(token_k, len(self.tokenizer))

        trie = Trie()
        for cand in all_cands:
            trie.insert([self._token2id(t) for t in self.tokenizer.tokenize(cand)])
        pending_node = trie.root

        best_prob, best_score = 0, 0

        step = 0
        while True:
            if pending_node.is_end_of_sequence and pending_node.is_valid:
                return decoding_ids, past_key_values, best_prob, best_score
            children = pending_node.get_children()

            outputs = self._model_forward(decoding_ids, attention_mask, past_key_values, encoder_outputs)
            logits = outputs.logits[:,-1,:].view(-1)
            
            tau = torch.ones_like(logits, device=self.device)
            tau[[node.key for node in children]] = 1 / temperature
            logits *= tau
            
            probs = torch.softmax(logits, -1)
            log_probs = torch.log(probs)
            
            children.sort(key=lambda child: probs[child.key].item(), reverse=True)
            for child in children[:5]:
                logging.info('\t' * step + f"token: {self._id2token(child.key)}, prob: {probs[child.key].item()}")
            if token_k:
                topk_probs, topk_idxs = probs.topk(token_k, -1, True, True)
                topk_idxs = {_idx.item() for _idx in topk_idxs}
                topk_tokens_with_probs = [(self._id2token(_idx.item()), round(_prob.item(), 4)) for _prob, _idx in zip(topk_probs, topk_idxs)]
                topk_tokens_with_probs.sort(key=lambda x: x[-1], reverse=True)
            
            pending_node = children[0]
            if probs[pending_node.key].item() < token_threshold or (token_k and pending_node.key not in topk_idxs):
                logging.info("\t\tNo valid candidates")
                return None, None, None, None
            decoding_ids = torch.cat([decoding_ids, torch.tensor([[pending_node.key]], dtype=torch.long, device=self.device)], -1)
            past_key_values = outputs.past_key_values
            best_prob = probs[pending_node.key].item()
            best_score += log_probs[pending_node.key].item()
            step += 1
    
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