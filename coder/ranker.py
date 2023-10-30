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
from coder.info_nce import InfoNCE
from coder.utils.trie import Trie
from coder.utils.metric_utils import Bleu, CodeBleu
from coder.constants import *

torch.manual_seed(42)  # pytorch random seed



class Generator:
    def __init__(
        self,
        model_name="Salesforce/codet5-base",
        checkpoint=None,
        device="cuda"
    ):
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.to(device)
        self.device = device
    
    def train(
        self,
        masked_codes,
        expected_types,
        epochs=5,
        train_batch_size=8,
        valid_batch_size=16,
        valid_ratio=0.2,
        learning_rate=1e-5,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=500,
        valid_step=5000,
        save_dir=None,
        best_k=3,

    ):
        all_examples = list(zip(masked_codes, expected_types))
        random.shuffle(all_examples)
        sep_idx = int(len(all_examples) * (1 - valid_ratio))
        train_examples, valid_examples = all_examples[:sep_idx], all_examples[sep_idx:]
        num_training_batchs = math.ceil(len(train_examples) / train_batch_size)
        num_training_steps = epochs * num_training_batchs

        logging.info(f"train examples: {len(train_examples)}")
        logging.info(f"valid examples: {len(valid_examples)}")
        logging.info(f"epochs: {epochs}")
        logging.info(f"learning rate: {learning_rate}")
        logging.info(f"train batch size: {train_batch_size}")
        logging.info(f"valid batch size: {valid_batch_size}")
        logging.info(f"train batch num: {num_training_batchs}")
        logging.info(f"valid step num: {num_training_steps}")
        logging.info(f"adam epsilon: {adam_epsilon}")
        logging.info(f"weight decay: {weight_decay}")
        logging.info(f"warmup steps: {warmup}")
        logging.info(f"max grad norm: {max_grad_norm}")
        logging.info(f"save dir: {save_dir}")
        logging.info(f"log step: {log_step}")

        logging.info("")
        logging.info("")

        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model_parameters, lr=learning_rate, eps=adam_epsilon)

        if warmup < 1:
            num_warmup_steps = num_training_steps * warmup
        else:
            num_warmup_steps = int(warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        total_steps = 0
        best_ckpts = []
        valid_step = num_training_batchs if valid_step == 0 else valid_step
        for cur_epoch in range(epochs):
            self.model.train()
            random.shuffle(train_examples)
            train_steps, train_loss = 0, 0
            batch_ranges = list(zip(range(0, len(train_examples), train_batch_size), range(train_batch_size, len(train_examples)+train_batch_size, train_batch_size)))
            batch_ranges = tqdm(batch_ranges, desc="Training", ascii=True)
            for beg, end in batch_ranges:
                total_steps += 1
                train_steps += 1
                batch = train_examples[beg:end]
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets = zip(*batch)
                sources = [f"{truncate_source(source, self.generator_tokenizer)}" for source in sources]
                targets = [f"{MASK}{target}" for target in targets]

                source_ids = self.generator_tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                target_ids = self.generator_tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                target_ids = target_ids.to(self.device)

#                 target_ids[target_ids == self.generator_tokenizer.pad_token_id] = -100

                # y_ids = target_ids[:, :-1].contiguous()
                # labels = target_ids[:, 1:].clone().detach()
                # labels[target_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

                attention_mask = source_ids.ne(self.generator_tokenizer.pad_token_id)
                decoder_attention_mask = target_ids.ne(self.generator_tokenizer.pad_token_id)
                outputs = self.model(
                    input_ids=source_ids,
                    attention_mask=attention_mask,
                    # decoder_input_ids=y_ids,
                    # labels=labels,
                    labels=target_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )

                loss = outputs.loss   
                train_loss += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if total_steps % log_step == 0 or total_steps % num_training_batchs == 0 or total_steps == num_training_steps:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epochs}, Batch {train_steps}/{len(batch_ranges)},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_training_steps:
                    bleu = self.evaluate_generator(valid_examples)
                    logging.info(f"[Validation] Step {total_steps}: bleu-4 {round(bleu, 4)}")

                    if save_dir is None:
                        continue
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    if len(best_ckpts) < best_k:
                        model_checkpoint = f"{save_dir}/generator-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts.append((model_checkpoint, bleu))
                    elif bleu > best_ckpts[-1][-1]:
                        os.unlink(best_ckpts[-1][0])
                        model_checkpoint = f"{save_dir}/generator-step{total_steps}.ckpt"
                        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts[-1] = (model_checkpoint, bleu)

                    best_ckpts.sort(key=lambda ckpt: ckpt[-1], reverse=True)
                    logging.info(f"Best checkpoints: {best_ckpts}")
                    # best_ckpt = best_ckpts[0][0]
                    json.dump(best_ckpts, open(f"{save_dir}/generator-ckpts.json", "w"), indent=4)
        del self.model

    def evaluate_generator(self, eval_examples, valid_batch_size=16, gen_max_len=10):
        self.model.eval()
        predictions, expectations = [], []
        batch_ranges = list(zip(range(0, len(eval_examples), valid_batch_size), range(valid_batch_size, len(eval_examples)+valid_batch_size, valid_batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Validation"):
                batch = eval_examples[beg:end]
                sources, targets = zip(*batch)
                expectations.extend(targets)

                sources = [f"{truncate_source(source, self.generator_tokenizer)}" for source in sources]
                targets = [f"{MASK}{target}" for target in targets]
                source_ids = self.generator_tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                attention_mask = source_ids.ne(self.generator_tokenizer.pad_token_id)
                outputs = self.model.generate(
                    source_ids,
                    attention_mask=attention_mask,
                    max_length=gen_max_len,
                    repetition_penalty=2.5,
                    num_beams=2
                )
                outputs = [self.generator_tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
                predictions.extend(outputs)

        predictions = [self.generator_tokenizer.tokenize(item.strip()) for item in predictions]
        expectations = [self.generator_tokenizer.tokenize(item.strip()) for item in expectations]

        bleu, *_ = Bleu.compute_bleu(expectations, predictions, smooth=True)
        return bleu
    
    def build_ranker_data(
        self,
        masked_codes,
        expected_types,
        user_types_list,
        k=5,
        max_len=25,
        batch_size=32,
        bleu_thres=0.2,
    ):
        self.model.eval()
        data_size = len(masked_codes)
        all_examples = []
        for beg, end in tqdm(list(zip(range(0, data_size, batch_size), range(batch_size, data_size + batch_size, batch_size))), desc="Building Data", ascii=True):
            batch_masked_codes = [truncate_source(source, self.generator_tokenizer) for source in masked_codes[beg:end]]
            batch_user_types_list = user_types_list[beg:end]
            batch_expected_types = expected_types[beg:end]
            input_ids = self.generator_tokenizer(batch_masked_codes, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_length=max_len, num_beams=k, num_return_sequences=k, return_dict_in_generate=True, output_scores=True)
            generations = self.generator_tokenizer.batch_decode(outputs.sequences)
            batch_gens_list = [generations[b:e] for b, e in zip(range(0, len(generations), k), range(k, len(generations) + k, k))]
            for code, gens, userdefs, tgt in zip(batch_masked_codes, batch_gens_list, batch_user_types_list, batch_expected_types):
                cleaned_gens = set()
                for gen in gens:
                    gen = gen.split("<extra_id_0>")[-1].split("<extra_id_1>")[0]
                    gen = self.generator_tokenizer.decode(self.generator_tokenizer.encode(gen), skip_special_tokens=True)
                    if gen.count("[") != gen.count("]"):
                        gen = gen.split("[")[0]
                    cleaned_gens.add(gen)
                
                cleaned_gens -= {tgt}
                userdefs = set() if userdefs is None else set(userdefs)
                userdefs -= {tgt}
                cleaned_gens = list(cleaned_gens)
                userdefs = list(userdefs)

                if len(cleaned_gens) == 0 and len(userdefs) == 0:
                    continue
                
                negs = list()
                if len(cleaned_gens) > 0:
                    neg = random.choice(cleaned_gens)
                    negs.append(neg)
                    cleaned_gens.remove(neg)
                rest_num = k - 1 - len(negs)
                cands = cleaned_gens + userdefs
                negs.extend(random.sample(cands, min(rest_num, len(cands))))
                while len(negs) < k - 1:
                    negs.append(random.choice(negs))
                all_examples.append((code, tgt, negs))
                # tgt_tokens = self.ranker_tokenizer.tokenize(tgt)
                # bleus = [Bleu.compute_bleu([tgt_tokens], [self.ranker_tokenizer.tokenize(neg)], max_order=2)[0] for neg in negs]
                # hard_neg, max_bleu = max(zip(negs, bleus), key=lambda p: p[-1])
                # if max_bleu < bleu_thres:
                #     continue
                # all_examples.append((code, tgt, hard_neg))
        return all_examples
    
    def train_ranker(
        self,
        all_examples,
        epochs=5,
        train_batch_size=8,
        valid_batch_size=16,
        valid_ratio=0.2,
        learning_rate=1e-5,
        # gen_k=5,
        # gen_max_len=30,
        # gen_batch_size=32,
        # bleu_thres=0.5,
        margin=0.2,
        adam_epsilon=1e-8,
        weight_decay=0,
        warmup=0.1,
        max_grad_norm=1.0,
        log_step=500,
        valid_step=5000,
        save_dir=None,
        best_k=3,

    ):  
        random.shuffle(all_examples)
        sep_idx = int(len(all_examples) * (1 - valid_ratio))
        train_examples, valid_examples = all_examples[:sep_idx], all_examples[sep_idx:]
        
        num_training_batchs = math.ceil(len(train_examples) / train_batch_size)
        num_training_steps = epochs * num_training_batchs

        neg_num = len(train_examples[0][-1])

        logging.info(f"epochs: {epochs}")
        logging.info(f"learning rate: {learning_rate}")
        logging.info(f"train batch size: {train_batch_size}")
        logging.info(f"valid batch size: {valid_batch_size}")
        # logging.info(f"generation k: {gen_k}")
        # logging.info(f"generation max length: {gen_max_len}")
        # logging.info(f"generation batch size: {gen_batch_size}")
        # logging.info(f"bleu threshold: {bleu_thres}")
        logging.info(f"margin: {margin}")
        logging.info(f"adam epsilon: {adam_epsilon}")
        logging.info(f"weight decay: {weight_decay}")
        logging.info(f"warmup steps: {warmup}")
        logging.info(f"max grad norm: {max_grad_norm}")
        logging.info(f"save dir: {save_dir}")
        logging.info(f"log step: {log_step}")
        logging.info("")
        logging.info("")
        logging.info(f"train examples: {len(train_examples)}")
        logging.info(f"valid examples: {len(valid_examples)}")
        logging.info(f"train batch num: {num_training_batchs}")
        logging.info(f"valid step num: {num_training_steps}")
        logging.info("")
        logging.info("")

        # frozen_params = set()
        # for name, param in self.ranker.encoder.named_parameters():
        #     param.requires_grad = False
        #     frozen_params.add(name)
        #     print(f"param: {name} is frozen...")
        # for block in self.ranker.decoder.block[:-1]:
        #     for name, param in block.named_parameters():
        #         param.requires_grad = False
        #         frozen_params.add(name)
        #         print(f"param: {name} is frozen...")
        # active_params = [(n, p) for n, p in self.ranker.named_parameters() if n not in frozen_params]
        active_params = [(n, p) for n, p in self.ranker.named_parameters()]
        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in active_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in active_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(model_parameters, lr=learning_rate, eps=adam_epsilon)

        if warmup < 1:
            num_warmup_steps = num_training_steps * warmup
        else:
            num_warmup_steps = int(warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        info_nce = InfoNCE(negative_mode='paired')

        total_steps = 0
        best_ckpts = []
        valid_step = num_training_batchs if valid_step == 0 else valid_step
        for cur_epoch in range(epochs):
            self.ranker.train()
            random.shuffle(train_examples)
            train_steps, train_loss = 0, 0
            batch_ranges = list(zip(range(0, len(train_examples), train_batch_size), range(train_batch_size, len(train_examples)+train_batch_size, train_batch_size)))
            batch_ranges = tqdm(batch_ranges, desc="Training", ascii=True)
            for beg, end in batch_ranges:
                total_steps += 1
                train_steps += 1
                batch = train_examples[beg:end]
                # descs, signatures, bodies = zip(*batch)
                # sources = self.pack_desc(descs)
                # targets = self.pack_code(signatures, bodies)

                sources, targets, negatives_list = zip(*batch)
                sources = [f"{truncate_source(source, self.ranker_tokenizer)}" for source in sources]
                targets = [f"{MASK}{target}" for target in targets]
                negatives_list = [[f"{MASK}{neg}" for neg in negs] for negs in negatives_list]

                source_ids = self.ranker_tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                source_ids = source_ids.to(self.device)
                target_ids = self.ranker_tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                target_ids = target_ids.to(self.device)
                
                attention_mask = source_ids.ne(self.ranker_tokenizer.pad_token_id)
                decoder_attention_mask = target_ids.ne(self.ranker_tokenizer.pad_token_id)
                outputs = self.ranker(
                    input_ids=source_ids,
                    attention_mask=attention_mask,
                    # decoder_input_ids=y_ids,
                    # labels=labels,
                    labels=target_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=True
                )
                logging.debug(outputs.encoder_last_hidden_state.shape)
                hidden_size = outputs.encoder_last_hidden_state.shape[-1]
                encoder_features = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                logging.debug(encoder_features.shape)
                length = torch.sum(attention_mask, axis=1) - 1
                encoder_features = torch.sum(encoder_features, axis=1)
                encoder_features = encoder_features * (1 / length.unsqueeze(-1))

                target_decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                logging.debug(target_decoder_features.shape)
                length = torch.sum(decoder_attention_mask, axis=1) - 1
                target_decoder_features = torch.sum(target_decoder_features, axis=1)
                target_decoder_features = target_decoder_features * (1 / length.unsqueeze(-1))
                # target_cos = torch.cosine_similarity(encoder_features, decoder_features, dim=-1)
                # logging.debug(target_cos.shape)
                
                negative_decoder_features = []
                for source, negatives in zip(sources, negatives_list):
                    _sources = [source] * neg_num
                    source_ids = self.ranker_tokenizer(_sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    negative_ids = self.ranker_tokenizer(negatives, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    negative_ids = negative_ids.to(self.device)

                    attention_mask = source_ids.ne(self.ranker_tokenizer.pad_token_id)
                    decoder_attention_mask = negative_ids.ne(self.ranker_tokenizer.pad_token_id)
                    outputs = self.ranker(
                        input_ids=source_ids,
                        attention_mask=attention_mask,
                        # decoder_input_ids=y_ids,
                        # labels=labels,
                        labels=negative_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        output_hidden_states=True
                    )
                    decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    logging.debug(decoder_features.shape)
                    length = torch.sum(decoder_attention_mask, axis=1) - 1
                    decoder_features = torch.sum(decoder_features, axis=1)
                    decoder_features = decoder_features * (1 / length.unsqueeze(-1))
                    negative_decoder_features.append(decoder_features)
                negative_decoder_features = torch.stack(negative_decoder_features, dim=0)
                
                loss = info_nce(encoder_features, target_decoder_features, negative_decoder_features)

                # loss = F.margin_ranking_loss(
                #     target_cos,
                #     negative_cos,
                #     torch.ones(target_cos.size(),
                #     device=target_cos.device),
                #     margin=margin,
                #     reduction="mean"
                # )
    
                train_loss += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ranker.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if total_steps % log_step == 0 or total_steps % num_training_batchs == 0 or total_steps == num_training_steps:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epochs}, Batch {train_steps}/{len(batch_ranges)},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_training_steps:
                    top5_acc = self.evaluate_ranker(valid_examples)
                    logging.info(f"[Validation] Step {total_steps}: top5 acc {round(top5_acc, 4)}")

                    if save_dir is None:
                        continue
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())
                    if len(best_ckpts) < best_k:
                        model_checkpoint = f"{save_dir}/ranker-step{total_steps}.ckpt"
                        model_to_save = self.ranker.module if hasattr(self.ranker, 'module') else self.ranker
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts.append((model_checkpoint, top5_acc))
                    elif top5_acc > best_ckpts[-1][-1]:
                        os.unlink(best_ckpts[-1][0])
                        model_checkpoint = f"{save_dir}/ranker-step{total_steps}.ckpt"
                        model_to_save = self.ranker.module if hasattr(self.ranker, 'module') else self.ranker
                        torch.save(model_to_save.state_dict(), model_checkpoint)
                        logging.info("Save the latest model into %s", model_checkpoint)
                        best_ckpts[-1] = (model_checkpoint, top5_acc)

                    best_ckpts.sort(key=lambda ckpt: ckpt[-1], reverse=True)
                    logging.info(f"Best checkpoints: {best_ckpts}")
                    # best_ckpt = best_ckpts[0][0]
                    json.dump(best_ckpts, open(f"{save_dir}/ranker-ckpts.json", "w"), indent=4)
        del self.ranker
    
    def evaluate_ranker(self, eval_examples, batch_size=48):
        self.ranker.eval()
        predictions = []
        with torch.no_grad():
            for source, target, negs in tqdm(eval_examples, ascii=True, desc="Validation"):
                cands = [target] + [neg for neg in negs if neg != target]
                rankings = []
                for beg, end in zip(range(0, len(cands), batch_size), range(batch_size, len(cands) + batch_size, batch_size)):
                    batch_cands = cands[beg:end]
                    sources = [truncate_source(source, self.ranker_tokenizer)] * len(batch_cands)
                    targets = [f"{MASK}{cand}" for cand in batch_cands]
                    source_ids = self.ranker_tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    target_ids = self.ranker_tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    target_ids = target_ids.to(self.device)
#                     target_ids[target_ids == self.ranker_tokenizer.pad_token_id] = -100
                    attention_mask = source_ids.ne(self.ranker_tokenizer.pad_token_id)
                    decoder_attention_mask = target_ids.ne(self.ranker_tokenizer.pad_token_id)
                    outputs = self.ranker(
                        input_ids=source_ids,
                        attention_mask=attention_mask,
                        # decoder_input_ids=y_ids,
                        # labels=labels,
                        labels=target_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        output_hidden_states=True
                    )
                    hidden_size = outputs.encoder_last_hidden_state.shape[-1]
                    encoder_features = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    length = torch.sum(attention_mask, axis=1) - 1
                    encoder_features = torch.sum(encoder_features, axis=1)
                    encoder_features = encoder_features * (1 / length.unsqueeze(-1))

                    decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    length = torch.sum(decoder_attention_mask, axis=1) - 1
                    decoder_features = torch.sum(decoder_features, axis=1)
                    decoder_features = decoder_features * (1 / length.unsqueeze(-1))
                    target_cos = torch.cosine_similarity(encoder_features, decoder_features)
                    cand_scores = [float(l) for l in target_cos.cpu().numpy().tolist()]
                    rankings.extend([(cand, l) for cand, l in zip(cands, cand_scores)])
                rankings.sort(key=lambda x:x[-1], reverse=True)
                r = [cand for cand, _ in rankings].index(target) + 1
                predictions.append(r)
        top5_acc = len([r for r in predictions if r <= 5]) / len(predictions)
        return top5_acc


    def predict(self, masked_codes, user_types_list, alpha=0.5, gen_k=5, gen_max_len=10, tem=1, gen_batch_size=12, rank_batch_size=64):
        self.model.eval()
        self.ranker.eval()
        data_size = len(masked_codes)
        original_predictions = []
        ranking_predictions = []
        for gen_beg, gen_end in tqdm(list(zip(range(0, data_size, gen_batch_size), range(gen_batch_size, data_size + gen_batch_size, gen_batch_size))), desc="Testing", ascii=True):
            batch_masked_codes = [truncate_source(source, self.generator_tokenizer) for source in masked_codes[gen_beg:gen_end]]
            batch_user_types_list = user_types_list[gen_beg:gen_end]
            input_ids = self.generator_tokenizer(batch_masked_codes, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.device)
            outputs = self.model.generate(input_ids, max_length=gen_max_len, num_beams=gen_k, num_return_sequences=gen_k, return_dict_in_generate=True, output_scores=True)
            
            generations = self.generator_tokenizer.batch_decode(outputs.sequences)
            scores = [float(s) for s in outputs.sequences_scores.cpu().numpy().tolist()]
#             scores = [0.] * len(generations)
            batch_gens_list = [list(zip(generations[b:e], scores[b:e])) for b, e in zip(range(0, len(generations), gen_k), range(gen_k, len(generations) + gen_k, gen_k)) ]

            for masked_code, gens, user_types in zip(batch_masked_codes, batch_gens_list, batch_user_types_list):
                logging.debug("----- masked code -----")
                logging.debug(masked_code)

                cleaned_gens = list()
                for gen, score in gens:
                    gen = gen.split("<extra_id_0>")[-1].split("<extra_id_1>")[0]
                    gen = self.generator_tokenizer.decode(self.generator_tokenizer.encode(gen), skip_special_tokens=True)
                    if gen.count("[") != gen.count("]"):
                        gen = gen.split("[")[0]
                    score = np.exp(score)
                    cleaned_gens.append((gen, score))
                    
                logging.debug(f"generations: {cleaned_gens}")
                logging.debug(f"user types: {user_types}")
                original_predictions.append(cleaned_gens)
                if user_types is None or len(user_types) == 0:
                    ranking_predictions.append([(gen, score, score, score) for gen, score in cleaned_gens])
                    if len(ranking_predictions) % 1000 == 0 or len(ranking_predictions) == data_size:
                        print(f"progress: {len(ranking_predictions)}/{data_size}")
                    continue
                
                # remove the last one to benefit user-defined types
                if len(cleaned_gens) == gen_k:
                    cleaned_gens = cleaned_gens[:-1]

                cands = {gen for gen, _ in cleaned_gens}
                cands.update(user_types)
                # for gen in cleaned_gens:
                #     cands.update(_mutate(gen, user_types))
                # print(cands)
                cands = list(cands)

                rankings = []
                for rank_beg, rank_end in zip(range(0, len(cands), rank_batch_size), range(rank_batch_size, len(cands) + rank_batch_size, rank_batch_size)):
                    batch_cands = cands[rank_beg:rank_end]
                    sources = [masked_code] * len(batch_cands)
                    targets = [f"{MASK}{cand}" for cand in batch_cands]

                    source_ids = self.ranker_tokenizer(sources, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    source_ids = source_ids.to(self.device)
                    target_ids = self.ranker_tokenizer(targets, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
                    target_ids = target_ids.to(self.device)

                    attention_mask = source_ids.ne(self.ranker_tokenizer.pad_token_id)
                    decoder_attention_mask = target_ids.ne(self.ranker_tokenizer.pad_token_id)
                    
                    with torch.no_grad():
                        outputs = self.ranker(input_ids=source_ids, attention_mask=attention_mask, labels=target_ids, decoder_attention_mask=decoder_attention_mask, output_hidden_states=True) 

                    hidden_size = outputs.encoder_last_hidden_state.shape[-1]
                    encoder_features = outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    length = torch.sum(attention_mask, axis=1) - 1
                    encoder_features = torch.sum(encoder_features, axis=1)
                    encoder_features = encoder_features * (1 / length.unsqueeze(-1))
                    decoder_features = outputs.decoder_hidden_states[-1] * decoder_attention_mask.unsqueeze(2).expand(-1,-1,hidden_size)
                    length = torch.sum(decoder_attention_mask, axis=1) - 1
                    decoder_features = torch.sum(decoder_features, axis=1)
                    decoder_features = decoder_features * (1 / length.unsqueeze(-1))
                    similarities = torch.cosine_similarity(encoder_features, decoder_features)
                    similarities = [float(l) for l in similarities.cpu().numpy().tolist()]

                    with torch.no_grad():
                        outputs = self.model(input_ids=source_ids, attention_mask=attention_mask, labels=target_ids, decoder_attention_mask=decoder_attention_mask, output_hidden_states=True)
                    
                    # logits = F.softmax(outputs.logits / tem, -1)
                    # target_logits = torch.gather(logits, dim=2, index=target_ids.unsqueeze(-1))
                    # target_logits = target_logits.squeeze(-1)
                    # target_logits = target_logits * decoder_attention_mask
                    # likelihoods = target_logits.sum(dim=-1) / decoder_attention_mask.sum(dim=-1)
                    # likelihoods = [float(l) for l in likelihoods.cpu().numpy().tolist()]
                    
                    logits = F.log_softmax(outputs.logits / tem, -1)
                    target_logits = torch.gather(logits, dim=2, index=target_ids.unsqueeze(-1))
                    target_logits = target_logits.squeeze(-1)
                    target_logits = target_logits * decoder_attention_mask
                    likelihoods = target_logits.sum(dim=-1)
                    likelihoods = torch.exp(likelihoods) 
                    likelihoods = [float(l) for l in likelihoods.cpu().numpy().tolist()]                   
                    
                    scores = [alpha * s + (1 - alpha) * l for s, l in zip(similarities, likelihoods)]
                    rankings.extend(list(zip(batch_cands, similarities, likelihoods, scores)))
                    
#                     print(target_ids[:, 0])
#                     logits = outputs.logits[:, 2:, :].clone()
#                     logits[:, 0] = -1e20
#                     logits[:, 0, target_ids[:, 0]] = outputs.logits[:, 2, target_ids[:, 0]]
#                     print(outputs.logits[0, 2, target_ids[:, 0]])
#                     print(logits[0, 0, target_ids[:, 0]])
#                     probs = F.softmax(logits / tem, -1)
#                     target_probs = torch.gather(probs, dim=2, index=target_ids.unsqueeze(-1))
#                     target_probs = target_probs.squeeze(-1)
#                     target_probs = target_probs * decoder_attention_mask
#                     for ids, _probs in zip(target_ids, target_probs):
#                         print([(self.tokenizer._convert_id_to_token(t.item()), p.item()) for t, p in zip(ids, _probs)])

                rankings.sort(key=lambda x:x[-1], reverse=True)
                logging.debug(f"rankings: {rankings}")
                ranking_predictions.append(rankings)
                if len(ranking_predictions) % 1000 == 0 or len(ranking_predictions) == data_size:
                    print(f"progress: {len(ranking_predictions)}/{data_size}")
        return original_predictions, ranking_predictions