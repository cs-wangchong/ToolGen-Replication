import ast
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
from torch.optim import AdamW
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

from coder.data_augmentor import DataAugmentor

from .utils.metric_utils import Bleu
from .constants import *

torch.manual_seed(42)  # pytorch random seed

class GeneratorTrainer:
    def __init__(
            self,
            model_name: str = "Salesforce/codet5-base",
            device="cuda",
        ):
        self.model_name = model_name
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.device = device
        self.model.to(self.device)

    def train(
            self,
            train_examples,
            valid_examples,
            epoch_num=2,
            train_batch_size=8,
            valid_batch_size=8,
            learning_rate=1e-4,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            max_grad_norm=1.0,
            train_sampling=None,
            valid_sampling=None,
            save_dir=None,
            log_step=500,
            valid_step=1000,
            best_k=3,
            max_len=512
        ):
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if train_sampling is not None and train_sampling < len(train_examples):
            train_examples = random.sample(train_examples, train_sampling)
        if valid_sampling is not None and valid_sampling < len(valid_examples):
            valid_examples = random.sample(valid_examples, valid_sampling)

        num_train_optimization_steps = epoch_num * math.ceil(len(train_examples) / train_batch_size)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

        if warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * warmup_steps
        else:
            warmup_steps = int(warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

        # Start training
        logging.info("***** Running training *****")
        logging.info(f"model name: {self.model_name}")
        
        logging.info(f"epoch num: {epoch_num}")
        logging.info(f"learning_rate: {learning_rate}")
        logging.info(f"adam epsilon: {adam_epsilon}")
        logging.info(f"weight decay: {weight_decay}")
        logging.info(f"warmup steps: {warmup_steps}")
        logging.info(f"max grad norm: {max_grad_norm}")
        logging.info(f"training sampling: {train_sampling}")
        logging.info(f"validation sampling: {valid_sampling}")
        logging.info(f"save dir: {save_dir}")
        logging.info(f"log step: {log_step}")
        logging.info(f"validation step: {valid_step}")

        logging.info("")
        logging.info("")
        logging.info(f"training examples: {len(train_examples)}")
        logging.info(f"validation examples: {len(valid_examples)}")
        logging.info(f"training batch size: {train_batch_size}")
        logging.info(f"validation batch size: {valid_batch_size}")
        logging.info("training batch num = %d", math.ceil(len(train_examples) / train_batch_size))
        logging.info("validation batch num = %d", math.ceil(len(valid_examples) / valid_batch_size))

        total_steps = 0
        best_ckpts = []
        for cur_epoch in range(epoch_num):
            self.model.train()
            random.shuffle(train_examples)
            train_steps, train_loss = 0, 0
            batch_ranges = list(zip(range(0, len(train_examples), train_batch_size), range(train_batch_size, len(train_examples)+train_batch_size, train_batch_size)))
            # bar = tqdm(list(zip(range(0, size, batch_size), range(batch_size, size+batch_size, batch_size))), desc="Training", ascii=True)
            for beg, end in batch_ranges:
                total_steps += 1
                train_steps += 1
                batch = train_examples[beg:end]
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets = zip(*batch)
                sources = [f"{source}" for source in sources]
                targets = [f"{target}" for target in targets]

                source_ids = self.tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                target_ids = self.tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                target_ids = target_ids.to(self.device)

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
                    output_hidden_states=True
                )
                loss = torch.mean(outputs.loss)
                train_loss += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if total_steps % log_step == 0 or total_steps == num_train_optimization_steps:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epoch_num}, Batch {train_steps}/{len(batch_ranges)},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_train_optimization_steps:
                    bleu = self.evaluate(valid_examples, valid_batch_size, max_len=max_len, f"{save_dir}/record-step{total_steps}.txt")
                    logging.info(f"[Validation] Step {total_steps}: bleu-4 {round(bleu, 4)}") 
            
                    if save_dir is None:
                        continue
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    if len(best_ckpts) < best_k:
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        # torch.save(model_to_save.state_dict(), f"{save_dir}/model-latest.bin")
                        logging.info("save the latest model into %s", model_checkpoint)
                        best_ckpts.append((model_checkpoint, bleu))
                    elif bleu > best_ckpts[-1][-1]:
                        os.unlink(best_ckpts[-1][0])
                        model_checkpoint = f"{save_dir}/model-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        # torch.save(model_to_save.state_dict(), f"{save_dir}/model-latest.bin")
                        logging.info("save the latest model into %s", model_checkpoint)
                        best_ckpts[-1] = (model_checkpoint, bleu)
                        
                    best_ckpts.sort(key=lambda ckpt: ckpt[-1], reverse=True)
                    logging.info(f"best checkpoints: {best_ckpts}")
                    best_ckpt = best_ckpts[0][0]
                    self.save_model_info(f"{save_dir}/model.json", best_ckpt)
        del self.model

    def evaluate(self, eval_set, batch_size=8, max_len=512, record_file=None):
        self.model.eval()
        queries, predictions, expectations = [], [], []
        batch_ranges = list(zip(range(0, len(eval_set), batch_size), range(batch_size, len(eval_set)+batch_size, batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Validation"):
                batch = eval_set[beg:end]
                # descs, codes = zip(*batch)
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets = zip(*batch)
                sources = [f"{source}" for source in sources]
                targets = [f"{target}" for target in targets]
                descs = sources

                outputs = self.generate(sources num_beams=2, num_return_sequences=1, max_len=max_len)
                queries.extend(descs)
                predictions.extend(outputs)
                expectations.extend(targets)
        
        if record_file is not None:
            data = []
            for query, pred, expt in zip(queries, predictions, expectations):
                d = f"query:{query}\nprediction:\n{pred}\nexpectation:\n{expt}"
                data.append(d)
            with Path(record_file).open("w") as f:
                f.write("\n\n".join(data))
                
        predictions = [self.tokenizer.tokenize(code.strip().replace(CODET5_LSP_POINT, "")) for code in predictions]
        expectations = [self.tokenizer.tokenize(code.strip().replace(CODET5_LSP_POINT, "")) for code in expectations]
                
        bleu, *_ = Bleu.compute_bleu(expectations, predictions, smooth=True)
        return bleu

    def generate(self, descs, num_beams=2, num_return_sequences=2, max_len=512):
        source_ids = self.tokenizer(descs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
        source_ids = source_ids.to(self.device)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            source_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            repetition_penalty=2.5, 
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            # length_penalty=1.0, 
            # early_stopping=True
        )
        outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
        if num_return_sequences > 1:
            outputs = [outputs[b:e] for b, e in zip(range(0, len(outputs), num_return_sequences), range(num_return_sequences, len(outputs)+num_return_sequences, num_return_sequences))]
        return outputs

    def save_model_info(self, path, checkpoint):
        json_data = {
            "model_name": self.model_name,
            "model_checkpoint": checkpoint,
        }
        with Path(path).open('w') as f:
            json.dump(json_data, f, indent=4)

