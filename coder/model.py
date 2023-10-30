import ast
import functools
import json
import logging
import time
import random
import math
from pathlib import Path
from typing import List
import re
import os
from collections import defaultdict

import jedi
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
from transformers.generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

from coder.data_augmentor import DataAugmentor
from coder.config import Config
from coder.utils.trie import Trie

from .utils.metric_utils import Bleu, CodeBleu
from .constants import *

torch.manual_seed(42)  # pytorch random seed

class Model:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer: RobertaTokenizer = None
        self.model: T5ForConditionalGeneration = None
        self.jedi_pj = None
        self.lsp_id = None

    def init_model(self, checkpoint=None):
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(self.config.model_name, use_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.config.device))
        self.model.to(self.config.device)
        self.lsp_id = self.tokenizer._convert_token_to_id(PLM_LSP_POINT)


    def train(
            self,
            train_examples,
            valid_examples,
            clean_lsp_points=False,
            data_parallel=False,
        ):
        if self.config.save_dir is not None:
            Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

        if self.model is None:
            self.init_model()

        if self.config.train_sampling is not None and self.config.train_sampling < len(train_examples):
            train_examples = random.sample(train_examples, self.config.train_sampling)
        if self.config.valid_sampling is not None and self.config.valid_sampling < len(valid_examples):
            valid_examples = random.sample(valid_examples, self.config.valid_sampling)

        num_train_batchs = math.ceil(len(train_examples) / self.config.train_batch_size)

        num_train_optimization_steps = self.config.epochs * num_train_batchs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

        if self.config.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * self.config.warmup_steps
        else:
            warmup_steps = int(self.config.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

        # Start training
        logging.info("***** Running training *****")
        logging.info("  Model name = %s", self.config.model_name)
        logging.info("  Training examples = %d", len(train_examples))
        logging.info("  Validation examples = %d", len(valid_examples))
        logging.info("  Training Batch size = %d", self.config.train_batch_size)
        logging.info("  Training Batch num = %d", math.ceil(len(train_examples) / self.config.train_batch_size))
        logging.info("  Validation Batch size = %d", self.config.valid_batch_size)
        logging.info("  Validation Batch num = %d", math.ceil(len(valid_examples) / self.config.valid_batch_size))
        
        logging.info("")
        logging.info("")
        logging.info(f"  Epoch: {self.config.epochs}")
        logging.info(f"  Learning rate: {self.config.learning_rate}")
        logging.info(f"  Adam epsilon: {self.config.adam_epsilon}")
        logging.info(f"  Weight decay: {self.config.weight_decay}")
        logging.info(f"  Warmup steps: {self.config.warmup_steps}")
        logging.info(f"  Max_grad_norm: {self.config.max_grad_norm}")
        logging.info(f"  Training sampling: {self.config.train_sampling}")
        logging.info(f"  Validation sampling: {self.config.valid_sampling}")
        logging.info(f"  Save dir: {self.config.save_dir}")
        logging.info(f"  Log step: {self.config.log_step}")
        logging.info(f"  Validation step: {self.config.valid_step}")

        if not self.config.valid_step:
            self.config.valid_step = num_train_batchs

        total_steps = 0
        topk_ckpts = [(None, None, 0)] * self.config.best_k
        bleu_decrese_count = 0

        dataloader = DataLoader(train_examples, batch_size=self.config.train_batch_size, shuffle=True)

        for cur_epoch in range(self.config.epochs):
            random.shuffle(train_examples)
            train_steps, train_loss = 0, 0
            # batch_ranges = list(zip(range(0, len(train_examples), self.config.train_batch_size), range(self.config.train_batch_size, len(train_examples)+self.config.train_batch_size, self.config.train_batch_size)))
            # bar = tqdm(list(zip(range(0, size, batch_size), range(batch_size, size+batch_size, batch_size))), desc="Training", ascii=True)
            for batch in dataloader:
                self.model.train()
                total_steps += 1
                train_steps += 1
                sources, targets = batch

                source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.config.device)

                if not clean_lsp_points:
                    target_ids = self.tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    target_ids = target_ids.to(self.config.device)
                else:
                    tokens_list = [[token.replace(PLM_LSP_POINT, "") for token in self.tokenizer.tokenize(target)] for target in targets]
                    ids_list = [
                        [self.tokenizer.bos_token_id] + [self.tokenizer._convert_token_to_id(token) for token in tokens] + [self.tokenizer.eos_token_id]
                        for tokens in tokens_list
                    ]
                    padding_len = max([len(ids) for ids in ids_list])
                    ids_list = [ids + [self.tokenizer.pad_token_id] * (padding_len - len(ids)) for ids in ids_list]
                    target_ids = torch.LongTensor(ids_list)
                    target_ids = target_ids.to(self.config.device)

                # y_ids = target_ids[:, :-1].contiguous()
                # labels = target_ids[:, 1:].clone().detach()
                # labels[target_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

                attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
                decoder_attention_mask = target_ids.ne(self.tokenizer.pad_token_id)
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=attention_mask, 
                    # decoder_input_ids=y_ids,
                    # labels=labels,
                    labels=target_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    # output_hidden_states=True
                )
                loss = torch.mean(outputs.loss)
                train_loss += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if total_steps % self.config.log_step == 0 or total_steps % num_train_batchs == 0:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{self.config.epochs}, Batch {train_steps}/{num_train_batchs},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % self.config.valid_step == 0 or total_steps == num_train_optimization_steps:
                    record_file = f"{self.config.save_dir}/record-step{total_steps}.txt"
                    bleu, codebleu = self.evaluate(valid_examples, record_file)
                    logging.info(f"[Validation] Step {total_steps}: bleu-4 {round(bleu, 4)}, codebleu {round(codebleu, 4)}") 
            
                    if self.config.save_dir is None:
                        continue
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    best_ckpt, _, best_bleu = topk_ckpts[0]
                    if codebleu > best_bleu:
                        last_ckpt, last_record, _ = topk_ckpts[0]
                        if last_ckpt:
                            os.unlink(last_ckpt)
                            os.unlink(last_record)
                        model_checkpoint = f"{self.config.save_dir}/model-step{total_steps}.ckpt"
                        topk_ckpts = [(model_checkpoint, record_file, codebleu)] + topk_ckpts[:-1]
                        best_ckpt = model_checkpoint
                        bleu_decrese_count = 0
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        # torch.save(model_to_save.state_dict(), f"{self.config.save_dir}/model-latest.bin")
                        logging.info("Save the latest model into %s", model_checkpoint)
                        
                    else:
                        bleu_decrese_count += 1
                        logging.info(f"NOTE: CodeBleu does not increase for {bleu_decrese_count} validations")
                        
                    logging.info(f"Top-{self.config.best_k} checkpoints: {topk_ckpts}")
                    self.save(f"{self.config.save_dir}/model.json", best_ckpt)

                    if bleu_decrese_count > self.config.patience:
                        break
            else:
                continue
            break
        del self.model

    def evaluate(self, eval_set, record_file=None):
        self.model.eval()
        f = Path(record_file).open("w") if record_file else None
        queries, predictions, expectations = [], [], []
        batch_ranges = list(zip(range(0, len(eval_set), self.config.valid_batch_size), range(self.config.valid_batch_size, len(eval_set)+self.config.valid_batch_size, self.config.valid_batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Validation"):
                batch = eval_set[beg:end]
                sources, targets = list(zip(*batch))[-2:]
                sources = [f"{source}" for source in sources]
                targets = [f"{target}" for target in targets]
                descs = sources
                outputs = self.generate(sources)
                queries.extend(descs)
                predictions.extend(outputs)
                expectations.extend(targets)

                if f:
                    data = []
                    for query, pred, expt in zip(descs, outputs, targets):
                        d = f"query:{query}\nprediction:\n{pred}\nexpectation:\n{expt}\n\n"
                        data.append(d)
                    f.write("".join(data))
        predictions = [code.strip().replace(PLM_LSP_POINT, "") for code in predictions]
        expectations = [code.strip().replace(PLM_LSP_POINT, "") for code in expectations]
        codebleu = CodeBleu.compute_codebleu(expectations, predictions)

        predictions = [self.tokenizer.tokenize(code) for code in predictions]
        expectations = [self.tokenizer.tokenize(code) for code in expectations]
        bleu = Bleu.compute_bleu(expectations, predictions, smooth=True)
        
        return bleu, codebleu

    def generate(self, descs):
        source_ids = self.tokenizer(descs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        source_ids = source_ids.to(self.config.device)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            source_ids,
            attention_mask=attention_mask,
            max_length=self.config.max_len,
            repetition_penalty=self.config.repetition_penalty, 
            num_beams=self.config.beam_size,
            # length_penalty=1.0, 
            # early_stopping=True
        )
        outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
        return outputs

    def evaluate_with_lsp(self, eval_set, cand_num=3, lsp_threshold=0.8, token_threshold=0.2, token_k=5, temperature=0.6, record_file=None):
        self.model.eval()
        f = Path(record_file).open("w") if record_file else None
        queries, predictions, expectations = [], [], []
        batch_ranges = list(zip(range(0, len(eval_set), self.config.valid_batch_size), range(self.config.valid_batch_size, len(eval_set)+self.config.valid_batch_size, self.config.valid_batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Validation"):
                batch = eval_set[beg:end]
                # descs, codes = zip(*batch)
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)
                _queries, _predictions, _expectations = [], [], []
                for repo, file_path, desc, code in batch:
                    _queries.append(desc)
                    _expectations.append(code)
                    self.update_lsp_project(repo)
                    context, line, column = self.get_lsp_context(file_path, code)
                    if context is None:
                        _predictions.append("")
                        continue
                    logging.info(f"====================================")
                    logging.info(f"[EXPECTATION]\n{code.strip()}")
                    output = self.generate_with_lsp(desc, file_path, context, line, column, cand_num, lsp_threshold, token_threshold, token_k, temperature)
                    _predictions.append(output)
                queries.extend(_queries)
                predictions.extend(_predictions)
                expectations.extend(_expectations)
        
                if f:
                    data = []
                    for query, pred, expt in zip(_queries, _predictions, _expectations):
                        d = f"query:{query}\nprediction:\n{pred}\nexpectation:\n{expt}\n\n"
                        data.append(d)
                    f.write("".join(data))
                
        predictions = [code.strip().replace(PLM_LSP_POINT, "") for code in predictions]
        expectations = [code.strip().replace(PLM_LSP_POINT, "") for code in expectations]
        codebleu = CodeBleu.compute_codebleu(expectations, predictions)

        predictions = [self.tokenizer.tokenize(code) for code in predictions]
        expectations = [self.tokenizer.tokenize(code) for code in expectations]
        bleu = Bleu.compute_bleu(expectations, predictions, smooth=True)
        return bleu, codebleu
    
    def _init_decoding(self, source_ids, attention_mask):
        decoding_ids = torch.tensor([self.model.generation_config.decoder_start_token_id], dtype=torch.long, device=self.config.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids = source_ids,
                attention_mask = attention_mask,
                decoder_input_ids = decoding_ids.unsqueeze(0),
            )
        encoder_outputs = BaseModelOutput(
            last_hidden_state=outputs.encoder_last_hidden_state,
            hidden_states=None,
            attentions=None,
        )
        logits = outputs.logits[:,-1,:]
        log_probs = torch.log_softmax(logits, -1).view(-1)
        beam_scores, best_idx = log_probs.topk(self.config.beam_size, 0, True, True)
        beam_decoding_ids = []
        for i in range(self.config.beam_size):
            beam_decoding_ids.append(torch.cat([decoding_ids, best_idx[i:i+1]], -1))
        beam_past_key_values = [outputs.past_key_values for _ in range(self.config.beam_size)]
        beam_probs = torch.zeros_like(beam_scores, device=self.config.device)
        return encoder_outputs, beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores
    
    def _beam_advance(
            self,
            encoder_outputs,
            attention_mask,
            decoding_inputs,
            past_key_values,
            cur_score,
            logits_processor
        ):
        with torch.no_grad():
            outputs = self.model(
                encoder_outputs = encoder_outputs,
                attention_mask = attention_mask,
                decoder_input_ids = decoding_inputs.unsqueeze(0)[:,-1:],
                past_key_values=past_key_values
            )
        logits = outputs.logits[:,-1,:]
        probs = torch.softmax(logits, -1)
        log_probs = torch.log(probs)
        log_probs = logits_processor(decoding_inputs.unsqueeze(0), log_probs)
        probs = probs.view(-1)
        log_probs = log_probs.view(-1)
        flat_scores = torch.ones_like(log_probs) * cur_score + log_probs
        best_scores, best_idx = flat_scores.topk(self.config.beam_size, 0, True, True)
        best_probs = probs[best_idx]

        best_decoding_ids = [torch.cat([decoding_inputs, best_idx[i:i+1]], -1) for i in range(self.config.beam_size)]
        best_past_key_values = [outputs.past_key_values for i in range(self.config.beam_size)]
        return best_decoding_ids, best_past_key_values, best_probs, best_scores
    
    def _lsp_expand(
            self,
            all_cands,
            encoder_outputs,
            attention_mask,
            decoding_inputs,
            past_key_values,
            logits_processor,
            expand_batch_size=16,
            cand_num=3,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6
        ):
        
        lsp_decoder_input_ids = decoding_inputs.unsqueeze(0).repeat(1, 1)
        lsp_past_key_values =  past_key_values
        lsp_probs = [0]
        lsp_scores =  [0]

        trie = Trie()
        for cand in all_cands:
            trie.insert([self.tokenizer._convert_token_to_id(t) for t in self.tokenizer.tokenize(cand)])

        # calculate scores for candidates
        pending_nodes = [trie.root]
        all_cands_with_scores = []

        step = 0
        while True:
            new_pending_nodes = []
            children_list = []
            for i, node in enumerate(pending_nodes):
                if node.is_end_of_sequence and node.is_valid:
                    _decoding_ids = lsp_decoder_input_ids[i]
                    _past_key_values = [[x[i:i+1,:,:,:] for x in y] for y in lsp_past_key_values]
                    _score = lsp_scores[i]
                    _prob = lsp_probs[i]
                    cand = self.tokenizer.decode(_decoding_ids[-step:], skip_special_tokens=True)
                    all_cands_with_scores.append((cand, step, _decoding_ids, _past_key_values, _prob, _score))
                elif len(node.get_children()) > 0:
                    new_pending_nodes.append(node)
                    children_list.append(node.get_children())
            pending_nodes = new_pending_nodes
            step += 1
            if len(pending_nodes) == 0:
                break
            
            logits_list = []
            past_kvs_list = []
            step_size = lsp_decoder_input_ids.size(0)
            for beg in range(0, step_size, expand_batch_size):
                end = min(beg + expand_batch_size, step_size)
                with torch.no_grad():
                    tmp_encoder_outputs = BaseModelOutput(
                        last_hidden_state=encoder_outputs.last_hidden_state.repeat(end-beg, 1, 1),
                        hidden_states=None,
                        attentions=None,
                    )
                    tmp_attention_mask = attention_mask.repeat(end-beg, 1, 1)
                    outputs = self.model(
                        encoder_outputs = tmp_encoder_outputs,
                        attention_mask = tmp_attention_mask,
                        decoder_input_ids = lsp_decoder_input_ids[beg:end,-1:],
                        past_key_values = [[x[beg:end] for x in y] for y in lsp_past_key_values]
                    )
                logits_list.append(outputs.logits[:,-1,:])
                past_kvs_list.append(outputs.past_key_values)
            logits = torch.cat(logits_list, 0)
            lsp_past_key_values = [[torch.cat(xs, 0) for xs in zip(*ys)] for ys in zip(*past_kvs_list)]
            tau = torch.ones_like(logits, device=self.config.device)
            tau_x, tau_y = [], []
            for idx, children in enumerate(children_list):
                tau_x.extend([idx] * len(children))
                tau_y.extend([node.key for node in children])
            tau[tau_x, tau_y] = temperature
            logits /= tau
            probs = torch.softmax(logits, -1)
            logprobs = torch.log(probs)
            # probs = logits_processor(tmp_decoder_input_ids, probs)
            topk_probs, topk_idxs = probs.topk(token_k, -1, True, True)

            next_token_ids_list = []
            next_token_probs_list = []
            next_token_logprobs_list = []
            new_pending_nodes = []
            for k, (_children, _probs, _logprobs, _topk_probs, _topk_idxs) in enumerate(zip(children_list, probs, logprobs, topk_probs, topk_idxs)):
                topk_tokens_with_probs = [(self.tokenizer._convert_id_to_token(_idx.item()), round(_prob.item(), 4)) for _prob, _idx in zip(_topk_probs, _topk_idxs)]
                topk_tokens_with_probs.sort(key=lambda x: x[-1], reverse=True)
                
                next_token_ids, next_token_probs, next_token_logprobs = [], [], []
                for child_node in _children:
                    if _probs[child_node.key].item() < token_threshold or child_node.key not in {_idx.item() for _idx in _topk_idxs}:
                        pending_nodes[k].remove_child(child_node.key)
                        continue
                    # child_node.set_score(_probs[child_node.key].item())
                    next_token_ids.append(child_node.key)
                    next_token_probs.append(_probs[child_node.key].item())
                    next_token_logprobs.append(_logprobs[child_node.key].item())
                    new_pending_nodes.append(child_node)
                next_token_ids_list.append(next_token_ids)
                next_token_probs_list.append(next_token_probs)
                next_token_logprobs_list.append(next_token_logprobs)
            
            pending_nodes = new_pending_nodes
            total_num = sum(len(next_token_ids) for next_token_ids in next_token_ids_list)
            if total_num == 0:
                break

            new_lsp_decoder_input_ids = []
            for i, next_token_ids in enumerate(next_token_ids_list):
                if len(next_token_ids) == 0:
                    continue
                expanded_rows = lsp_decoder_input_ids[i:i+1].repeat(len(next_token_ids), 1)
                next_ids = torch.tensor([[t] for t in next_token_ids], dtype=torch.long, device=self.config.device)
                new_lsp_decoder_input_ids.append(torch.cat((expanded_rows, next_ids), -1))
            lsp_decoder_input_ids = torch.cat(new_lsp_decoder_input_ids, 0)   
            new_lsp_past_key_values = []
            for y in lsp_past_key_values:
                new_y = []
                for x in y:
                    new_x = []
                    for i, next_token_ids in enumerate(next_token_ids_list):
                        if len(next_token_ids) == 0:
                            continue
                        expanded_rows = x[i:i+1].repeat(len(next_token_ids), 1, 1, 1)
                        new_x.append(expanded_rows)
                    new_y.append(torch.cat(new_x, 0))
                new_lsp_past_key_values.append(new_y)
            lsp_past_key_values = new_lsp_past_key_values
            lsp_probs = [prob for next_token_probs in next_token_probs_list for prob in next_token_probs]
            lsp_scores = [score + logprob for score, next_token_logprobs in zip(lsp_scores, next_token_logprobs_list) for logprob in next_token_logprobs]
        
        if len(all_cands_with_scores) == 0:
            logging.info("\t\tNo valid candidates")
            return [], [], None, None
        all_cands_with_scores.sort(key=lambda x: x[-1], reverse=True)
        logging.info(f"\t\tCandidates with scores:")
        for r, (_cand, _len, _, _, _, _score) in enumerate(all_cands_with_scores, 1):
            logging.info(f"\t\t\tNo. {r}: {repr(_cand)}, score: {_score}, length: {_len}")
            # for _step, (cand_token, cand_prob, topk_tokens) in enumerate(cand2steps[_cand]):
            #     print(f"\t\t\tstep: {_step} | {cand_token} | {cand_prob} | {topk_tokens}")
        _, _, lsp_decoding_ids, lsp_past_key_values, lsp_probs, lsp_scores = zip(*all_cands_with_scores[:cand_num])
        lsp_probs = torch.tensor(lsp_probs, dtype=torch.float, device=self.config.device)
        lsp_scores = torch.tensor(lsp_scores, dtype=torch.float, device=self.config.device)
        # print(lsp_decoding_ids, lsp_past_key_values, lsp_probs, lsp_scores)
        return list(lsp_decoding_ids), list(lsp_past_key_values), lsp_probs, lsp_scores

    
    def _update_bank(self, bank, best_decoding_ids, best_past_key_values, best_probs, best_scores):
        # best_generations = [self.tokenizer.decode(decoding_ids[1:], skip_special_tokens=True) for decoding_ids in best_decoding_ids]
        if bank is None:
            bank = [best_decoding_ids, best_past_key_values, best_probs, best_scores]
            # bank.append((best_decoding_ids, best_past_key_values, best_probs, best_scores, best_generations))
        else:
            bank[0].extend(best_decoding_ids)
            bank[1].extend(best_past_key_values)
            bank[2] = torch.cat([bank[2], best_probs], 0)
            bank[3] = torch.cat([bank[3], best_scores], 0)
            # bank[4].extend(best_generations)
        return bank

    def _select_from_bank(self, bank):
        cand_decoding_ids, cand_past_key_values, cand_probs, cand_scores = bank
        beam_scores, best_idx = cand_scores.topk(self.config.beam_size, 0, True, True)
        beam_probs = cand_probs[best_idx]
        beam_decoding_ids = [cand_decoding_ids[best_idx[i].item()] for i in range(self.config.beam_size)]
        beam_past_key_values = [cand_past_key_values[best_idx[i].item()] for i in range(self.config.beam_size)]
        return beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores

    def generate_with_lsp(
            self,
            desc,
            file_path,
            code_context,
            line,
            column,
            cand_num=3,
            lsp_threshold=0.8,
            token_threshold=0.2,
            token_k=5,
            temperature=0.6
        ):
        source_ids = self.tokenizer([desc], add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        source_ids = source_ids.to(self.config.device)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)

        logits_processor = LogitsProcessorList()
        if self.config.repetition_penalty > 1:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=self.config.repetition_penalty))

        encoder_outputs, beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores = self._init_decoding(source_ids, attention_mask)

        final_generation_ids = []
        for step_idx in range(1, self.config.max_len):
            for beam_idx, decoding_ids in enumerate(beam_decoding_ids):
                if sum(decoding_ids.eq(self.tokenizer.eos_token_id).long()) >= 1: # type: ignore
                    # func = self.tokenizer.decode(decoding_ids[1:], skip_special_tokens=True).strip()
                    # if len(func) == 0:
                    #     continue
                    if sum(decoding_ids.eq(self.tokenizer.bos_token_id).long()) >= 1:
                        final_generation_ids.append((decoding_ids[1:], beam_scores[beam_idx].item()))
                    beam_scores[beam_idx] = -1e10
            if len(final_generation_ids) >= self.config.beam_size:
                break
            
            for beam_idx in range(len(beam_decoding_ids)):
                func = self.tokenizer.decode(beam_decoding_ids[beam_idx][1:], skip_special_tokens=True)
                if not func.endswith(PLM_LSP_POINT) or beam_probs[beam_idx].item() < lsp_threshold:
                    continue
                logging.info(f"[ENTER LSP] confidence: {beam_probs[beam_idx].item()} >= {lsp_threshold}")
                logging.info(f"\tcurrent code: ```\n{func}\n```")
                cands = self.get_lsp_completions(func, file_path, code_context, line, column)
                logging.info(f"\tcandidates by LSP: {cands}")
                if len(cands) == 0:
                    continue
                lsp_decoding_ids, lsp_past_key_values, lsp_probs, lsp_scores = self._lsp_expand(
                    cands,
                    encoder_outputs,
                    attention_mask,
                    beam_decoding_ids[beam_idx],
                    beam_past_key_values[beam_idx],
                    logits_processor,
                    cand_num=cand_num,
                    token_threshold=token_threshold,
                    token_k=token_k,
                    temperature=temperature
                )
                if len(lsp_decoding_ids) == 0:
                    continue
                
                # beam_decoding_ids.extend(lsp_decoding_ids)
                # beam_past_key_values.extend(lsp_past_key_values)
                # beam_probs = torch.cat([beam_probs, lsp_probs], 0)
                # # beam_scores = torch.cat([beam_scores, beam_scores[beam_idx:beam_idx+1].repeat(len(lsp_decoding_ids))], 0)
                # lsp_scores = beam_scores[beam_idx:beam_idx+1].repeat(len(lsp_decoding_ids)) + lsp_scores
                # beam_scores = torch.cat([beam_scores, lsp_scores], 0)

                beam_decoding_ids = beam_decoding_ids[:beam_idx] + beam_decoding_ids[beam_idx+1:] + lsp_decoding_ids
                beam_past_key_values = beam_past_key_values[:beam_idx] + beam_past_key_values[beam_idx+1:] + lsp_past_key_values
                beam_probs = torch.cat([beam_probs[:beam_idx], beam_probs[beam_idx+1:], lsp_probs], 0)
                lsp_scores = beam_scores[beam_idx:beam_idx+1].repeat(len(lsp_decoding_ids)) + lsp_scores
                beam_scores = torch.cat([beam_scores[:beam_idx], beam_scores[beam_idx+1:], lsp_scores], 0)

            bank = None
            for beam_idx in range(len(beam_decoding_ids)):
                best_decoding_ids, best_past_key_values, best_probs, best_scores = self._beam_advance(
                    encoder_outputs,
                    attention_mask,
                    beam_decoding_ids[beam_idx],
                    beam_past_key_values[beam_idx],
                    beam_scores[beam_idx],
                    logits_processor,
                )
                bank = self._update_bank(bank, best_decoding_ids, best_past_key_values, best_probs, best_scores)
            
            if not bank:
                break
            beam_decoding_ids, beam_past_key_values, beam_probs, beam_scores = self._select_from_bank(bank)

        # for l, (tmp_generated_ids, tmp_decoder_input_ids, tmp_past_key_values, tmp_scores) in bank.items():
        #     for ids, s in zip(tmp_generated_ids, tmp_scores):
        #         func = self.tokenizer.decode(ids[1:], skip_special_tokens=True)
        #         cands.append((func, s))
        if len(final_generation_ids) == 0:
            for i, decoding_ids in enumerate(beam_decoding_ids): 
                # func, score = self.tokenizer.decode(decoding_ids[1:], skip_special_tokens=True).strip(), beam_scores[i].item()
                # if len(func) == 0:
                #     continue
                final_generation_ids.append((decoding_ids[1:], beam_scores[i].item()))
        
        final_generation_ids = list(sorted(final_generation_ids, key=lambda pair:pair[-1], reverse=True))
        final_generations = []
        logging.info(f"[PREDICTIONS]")
        for idx, (ids, s) in enumerate(final_generation_ids, 1):
            logging.info(f"Prediction {idx}, socre: {s}")
            logging.info(f"\n{self.tokenizer.decode(ids, skip_special_tokens=False).strip()}")
            func = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            if len(func) == 0:
                continue
            final_generations.append((func, s))
        logging.info("")
        return final_generation_ids[0][0]
    
    def update_lsp_project(self, pj_path, py_env=None):
        if self.jedi_pj is None or self.jedi_pj._path != pj_path:
            self.jedi_pj: jedi.Project = jedi.Project(pj_path, environment_path=py_env, added_sys_path=(pj_path,))

    def get_lsp_context(self, file_path: str, func_code:str):
        with Path(file_path).open("r") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
        except Exception as e:
            return None, None, None
        funcs = (func for func in ast.walk(tree) if isinstance(func, ast.FunctionDef))
        
        for func in funcs:
            func_source = ast.get_source_segment(code, func)
            line, column = func.lineno, func.col_offset
            rest_source = code.replace(func_source, "<PLACEHOLDER>")
            _, tokens = DataAugmentor.tokenize_func(func_source)
            original_func = "".join(tokens).replace(PLM_LSP_POINT, "<unk>").strip()
            if original_func == func_code.replace(PLM_LSP_POINT, "").strip():
                return rest_source, line, column
        return None, None, None
    
    def get_lsp_completions(self, func, file_path, code_context, line, column):
        cleand_func = func.replace(PLM_LSP_POINT, "")
        lines = cleand_func.split("\n")
        _line, _column = line, column
        if len(lines) == 1:
            _column += len(lines[0])
        else:
            _line += len(lines) - 1
            _column = len(lines[-1])
        context = code_context.replace("<PLACEHOLDER>", cleand_func)
        try:
            script = jedi.Script(project=self.jedi_pj, path=file_path, code=context)
            completions = script.complete(line=_line, column=_column)
        except Exception as e:
            completions = []
        completions = [completion.complete.strip() for completion in completions]
        completions = [
            completion for completion in completions 
            if len(completion) > 0 and not completion.startswith("__") and completion not in SKIPPED
        ]
        return completions


    def save(self, path, checkpoint):
        json_data = {
            "config": self.config.to_json(),
            "model_checkpoint": checkpoint,
        }
        with Path(path).open('w') as f:
            json.dump(json_data, f, indent=4)

    @classmethod
    def load(cls, path, device="cuda"):
        with Path(path).open('r') as f:
            json_data = json.load(f)
        config = Config.from_json(json_data["config"])
        print(f"model name: {config.model_name}")
        
        config.device = device
        model = cls(config)
        model.init_model(checkpoint=json_data["model_checkpoint"])
        # retriever.init_vocab_model(checkpoint=json_data["vocab_model_checkpoint"])
        return model

