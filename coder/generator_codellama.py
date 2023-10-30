import json
import logging
import random
import math
from pathlib import Path
import re
import os
import shutil

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers import AddedToken
from transformers import get_linear_schedule_with_warmup, TrainingArguments, Trainer

from coder.utils.metric_utils import Bleu, CodeBleu
from coder.constants import *

torch.manual_seed(42)  # pytorch random seed

def clean_lsp(code:str):
    code = re.sub(r" ?%s ?" % re.escape(PLM_LSP_POINT), "", code)
    return code

def clean_str(code):
    code = re.sub(r"'(.*?)'", "", code)
    code = re.sub(r'"(.*?)"', "", code)
    return code.strip()

def reformat_prompt(prompt):
    desc, sign = prompt.split("Function Signature:")
    desc = desc[len("Description:"):]
    return f"### Description:\n{desc.strip()}\n\n### Signature:\n{sign.strip()}\n\n### Code:\n"

class CodeLlamaGenerator:
    def __init__(
            self,
            model_name="codellama/CodeLlama-7b-Python-hf",
            checkpoint=None,
            additional_tokens=[PLM_LSP_POINT],
            device="cuda"
        ):
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
        self.tokenizer.add_eos_token = True
        self.tokenizer.pad_token_id = 0
        # self.tokenizer.padding_side = "left"
        if additional_tokens is None:
            additional_tokens = []
        if len(additional_tokens) > 0:
            self.tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
            self.model_embeds = self.model.resize_token_embeddings(len(self.tokenizer))
        if checkpoint is not None:
            self.model = PeftModel.from_pretrained(self.model, checkpoint)
            if len(additional_tokens) > 0:
                additional_embeddings = torch.load(open(f"{checkpoint}/additional_embeddings.pt", "rb"))
                self.model_embeds.weight[-len(additional_tokens):] = additional_embeddings
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.enable_input_require_grads()
            self.model = prepare_model_for_int8_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            for name, params in self.model.named_parameters():
                if params.requires_grad:
                    print(name)
        self.model.to(device)
        self.model_name = model_name
        self.additional_tokens = additional_tokens
        self.device = device

    def train(
            self,
            train_examples,
            valid_examples,
            epoch_num=2,
            learning_rate=1e-4,
            train_batch_size=8,
            valid_batch_size=8,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            max_grad_norm=1.0,
            train_sampling=None,
            valid_sampling=None,
            best_k=3,
            patience=5,
            max_len=512,
            save_dir=None,
            log_step=500,
            valid_step=None,
            report_prediction=False
        ):
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        if train_sampling is not None and train_sampling < len(train_examples):
            train_examples = random.sample(train_examples, train_sampling)
        if valid_sampling is not None and valid_sampling < len(valid_examples):
            valid_examples = random.sample(valid_examples, valid_sampling)

        num_train_batchs = math.ceil(len(train_examples) / train_batch_size)

        num_train_optimization_steps = epoch_num * num_train_batchs
        # no_decay = ['bias', 'LayerNorm.weight']
        no_decay = []
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # print([n for n in self.model.named_parameters()])
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

        if warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * warmup_steps
        else:
            warmup_steps = int(warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

        # Start training
        logging.info("***** Running training *****")
        logging.info("  Model name = %s", self.model_name)
        logging.info("  Training examples = %d", len(train_examples))
        logging.info("  Validation examples = %d", len(valid_examples))
        logging.info("  Training Batch size = %d", train_batch_size)
        logging.info("  Training Batch num = %d", math.ceil(len(train_examples) / train_batch_size))
        logging.info("  Validation Batch size = %d", valid_batch_size)
        logging.info("  Validation Batch num = %d", math.ceil(len(valid_examples) / valid_batch_size))
        
        logging.info("")
        logging.info("")
        logging.info(f"  Epoch: {epoch_num}")
        logging.info(f"  Learning rate: {learning_rate}")
        logging.info(f"  Adam epsilon: {adam_epsilon}")
        logging.info(f"  Weight decay: {weight_decay}")
        logging.info(f"  Warmup steps: {warmup_steps}")
        logging.info(f"  Max_grad_norm: {max_grad_norm}")
        logging.info(f"  Training sampling: {train_sampling}")
        logging.info(f"  Validation sampling: {valid_sampling}")
        logging.info(f"  Save dir: {save_dir}")
        logging.info(f"  Log step: {log_step}")
        logging.info(f"  Validation step: {valid_step}")
        logging.info(f"  Best k: {best_k}")
        logging.info(f"  Patience: {patience}")
        logging.info(f"  Max len: {max_len}")
        logging.info(f"  Report prediction: {report_prediction}")
        logging.info("")
        logging.info("")

        if not valid_step:
            valid_step = num_train_batchs

        total_steps = 0
        topk_ckpts = [(None, None, 0)] * best_k
        decresing_count = 0

        dataloader = DataLoader(train_examples, batch_size=train_batch_size, shuffle=True)

        for cur_epoch in range(epoch_num):
            random.shuffle(train_examples)
            train_steps, train_loss = 0, 0
            for batch in tqdm(dataloader, total=num_train_batchs, desc=f"Epoch {cur_epoch+1}", ascii=True):
                self.model.train()
                total_steps += 1
                train_steps += 1
                prompts, codes = batch
                inputs = [f"{reformat_prompt(prompt)}{code}" for prompt, code in zip(prompts, codes)]

                input_ids = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").input_ids
                input_ids = input_ids.to(self.device)
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    labels=input_ids,
                    # output_hidden_states=True
                )
                loss = torch.mean(outputs.loss)
                train_loss += loss.item()

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if total_steps % log_step == 0 or total_steps % num_train_batchs == 0:
                    logging.info(f"[Training] Step {total_steps}, Epoch {cur_epoch+1}/{epoch_num}, Batch {train_steps}/{num_train_batchs},  Train loss {round(train_loss / train_steps, 6)}")

                if total_steps % valid_step == 0 or total_steps == num_train_optimization_steps:
                    record_file = f"{save_dir}/record-step{total_steps}.txt" if save_dir and report_prediction else None
                    bleu, codebleu, lsp_hit = self.evaluate(valid_examples, valid_batch_size, max_len, beam_size=1, record_file=record_file)
                    logging.info(f"[Validation] Step {total_steps}, bleu-4: {round(bleu, 4)}, codebleu: {round(codebleu, 4)}, lsp hit: {round(lsp_hit, 4)}") 
            
                    if save_dir is None:
                        continue
                    # timestamp = time.strftime("%m%d-%H%M", time.localtime())

                    best_ckpt, _, best_bleu = topk_ckpts[0]
                    if codebleu > best_bleu:
                        decresing_count = 0
                    else:
                        decresing_count += 1
                        logging.info(f"NOTE: CodeBleu does not increase for {decresing_count} validations")

                    last_ckpt, last_record, last_bleu = topk_ckpts[-1]
                    if codebleu > last_bleu:
                        if last_ckpt:
                            shutil.rmtree(last_ckpt)
                        if last_record:
                            os.unlink(last_record)
                        model_checkpoint = f"{save_dir}/model-step{total_steps}"
                        topk_ckpts = topk_ckpts[:-1] + [(model_checkpoint, record_file, codebleu)]
                        topk_ckpts.sort(key=lambda t:t[-1], reverse=True)
                        self.model.save_pretrained(model_checkpoint)
                        if len(self.additional_tokens) > 0:
                            torch.save(
                                self.model_embeds.weight[-len(self.additional_tokens):],
                                open(f"{model_checkpoint}/additional_embeddings.pt", "wb")
                            )
                        # torch.save(model_to_save.state_dict(), f"{save_dir}/model-latest.bin")
                        logging.info("Save the latest model into %s", model_checkpoint)
                    elif record_file:
                        os.unlink(record_file)

                    logging.info(f"Top-{best_k} checkpoints: {topk_ckpts}")
                    best_ckpt, _, best_bleu = topk_ckpts[0]
                    self.save_model_info(f"{save_dir}/model.json", best_ckpt)

                    if decresing_count > patience:
                        break
            else:
                continue
            logging.info(f"Early stop at {total_steps} steps")
            break

    def evaluate(self, eval_set, batch_size=8, max_len=512, beam_size=2, repetition_penalty=1.0, record_file=None):
        self.model.eval()
        queries, predictions, expectations = [], [], []
        batch_ranges = list(zip(range(0, len(eval_set), batch_size), range(batch_size, len(eval_set)+batch_size, batch_size)))
        with torch.no_grad():
            for beg, end in tqdm(batch_ranges, ascii=True, desc="Evaluation"):
                batch = eval_set[beg:end]

                prompts, codes = zip(*batch)
                inputs = [reformat_prompt(prompt) for prompt in prompts]
                outputs = self.generate(inputs, beam_size=beam_size, max_len=max_len, repetition_penalty=repetition_penalty)
                
                queries.extend(prompts)
                predictions.extend(outputs)
                expectations.extend(codes)
        
        total_count = 0
        hit_count = 0
        infos = []
        for query, pred, expt in zip(queries, predictions, expectations):
            info = f"======================\n[PROMPT]:\n{query}\n[EXPECTATION]:\n{expt}\n[PREDICTION]:\n{pred}\n"
            signature = query.split("Function Signature:")[-1].strip()
            key_eles = set()
            for mobj in re.finditer(r"(%s)(\w+)(\W|$)" % re.escape(PLM_LSP_POINT), expt):
                key_ele = mobj.group(2).strip()
                if len(key_ele) == 0:
                    continue
                if re.search(r"\s+%s\s*=" % re.escape(key_ele), expt):
                    continue
                if re.search(r"as\s+%s\s*:" % re.escape(key_ele), expt):
                    continue
                if re.search(r"for\s+\W*%s\W*\s+in" % re.escape(key_ele), expt):
                    continue
                if re.search(r"(\W|^)%s(\W|$)" % re.escape(key_ele), signature):
                    continue
                key_eles.add(key_ele)
            for key_ele in key_eles:
                total_count += 1
                if re.search(r"(\W|^)%s(\W|$)" % re.escape(key_ele), clean_str(pred)):
                    hit_count += 1
                    info = f"{info}(√) {key_ele}\n"
                else:
                    info = f"{info}(x) {key_ele}\n"
            infos.append(info)
        logging.info(f"total lsp count: {total_count}, hit lsp count: {hit_count}")
        lsp_hit = hit_count/total_count if total_count > 0 else 0

        predictions = [clean_lsp(code) for code in predictions]
        expectations = [clean_lsp(code) for code in expectations]
        codebleu = CodeBleu.compute_codebleu(expectations, predictions)        
        
        predictions = [self.tokenizer.tokenize(code) for code in predictions]
        expectations = [self.tokenizer.tokenize(code) for code in expectations]
        bleu4 = Bleu.compute_bleu(expectations, predictions, smooth=True)

        if record_file is not None:
            with Path(record_file).open("w") as f:
                f.write("\n".join(infos))

        return bleu4, codebleu, lsp_hit

    def generate(self, inputs, beam_size=2, max_len=512, repetition_penalty=1.0):
        input_ids = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").input_ids
        input_ids = input_ids[:,:-1] # remove </s>
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            repetition_penalty=repetition_penalty,
            num_beams=beam_size,
            num_return_sequences=1,
            # length_penalty=1.0, 
            # early_stopping=True
        )
        outputs = [self.tokenizer.decode(cand, skip_special_tokens=True) for cand in outputs]
        outputs = [out[len(inp):].strip() for (inp, out) in zip(inputs, outputs)]
        return outputs

    def save_model_info(self, path, checkpoint):
        json_data = {
            "model_name": self.model_name,
            "additional_tokens": self.additional_tokens,
            "model_checkpoint": checkpoint,
        }
        with Path(path).open('w') as f:
            json.dump(json_data, f, indent=4)

    @classmethod
    def load_from_model_info(cls, path, device="cuda"):
        with Path(path).open('r') as f:
            json_data = json.load(f)
        model = cls(json_data['model_name'], json_data['model_checkpoint'], json_data['additional_tokens'], device)
        return model