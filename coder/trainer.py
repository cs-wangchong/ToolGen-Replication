import logging
import math
from pathlib import Path
import random
import sys
from typing import List, Dict, Any, Callable

import torch
from peft import PeftModelForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers import PreTrainedModel
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

from coder.constants import *
from coder.prompt import *

torch.manual_seed(42)  # pytorch random seed



class DataCollator:
    def __init__(self, tokenizer:PreTrainedTokenizerBase, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }
        
        if self.tokenizer.padding_side == "left":
            max_length = max(len(item["input_ids"]) for item in features)
            max_length = (
                    (max_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for item in features:
                input_pad = [self.tokenizer.pad_token_id] * (max_length - len(item["input_ids"]))
                mask_pad = [0] * (max_length - len(item["attention_mask"]))
                label_pad = [-100] * (max_length - len(item["labels"]))
                batch["input_ids"].append(input_pad + item["input_ids"])
                batch["attention_mask"].append(mask_pad + item["attention_mask"])
                batch["labels"].append(label_pad + item["labels"])
        else:
            max_input_length = max(len(item["input_ids"]) for item in features)
            max_input_length = (
                    (max_input_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            max_label_length = max(len(item["labels"]) for item in features)
            max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for item in features:
                input_pad = [self.tokenizer.pad_token_id] * (max_input_length - len(item["input_ids"]))
                mask_pad = [0] * (max_input_length - len(item["attention_mask"]))
                label_pad = [-100] * (max_label_length - len(item["labels"]))
                batch["input_ids"].append(item["input_ids"] + input_pad)
                batch["attention_mask"].append(item["attention_mask"] + mask_pad)
                batch["labels"].append(item["labels"] + label_pad)
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["labels"] = torch.tensor(batch["labels"])
        return batch

def train(
        model:PreTrainedModel,
        tokenizer:PreTrainedTokenizer,
        train_examples,
        valid_examples,
        epoch_num=2,
        learning_rate=1e-4,
        weight_decay=0.05,
        train_batch_size=8,
        valid_batch_size=8,
        warmup_steps=0.1,
        save_limit=3,
        max_len=512,
        save_dir=None,
        logging_steps=500,
        valid_steps=None,
    ):

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    train_batchs = math.ceil(len(train_examples) / train_batch_size)
    max_steps = epoch_num * train_batchs
    if warmup_steps < 1:
        warmup_steps = int(max_steps * warmup_steps)
    else:
        warmup_steps = int(warmup_steps)
    if not valid_steps:
        valid_steps = train_batchs
    
    logging.info("***** Running training *****")
    logging.info("  Model name = %s", type(model).__name__)
    logging.info("  Additional tokens = %s", type(model).__name__)
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
    logging.info(f"  Weight decay: {weight_decay}")
    logging.info(f"  Warmup steps: {warmup_steps}")
    logging.info(f"  Save dir: {save_dir}")
    logging.info(f"  Logging steps: {logging_steps}")
    logging.info(f"  Validation steps: {valid_steps}")
    logging.info(f"  Save limit: {save_limit}")
    logging.info(f"  Max len: {max_len}")
    logging.info("")
    logging.info("")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=valid_batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epoch_num,
        warmup_steps=warmup_steps,
        # lr_scheduler_type = "constant",
        max_steps=max_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        logging_first_step=True,
        optim="adamw_torch",
        # optim="paged_adamw_32bit",
        # max_grad_norm=0.3,
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=valid_steps,
        save_steps=valid_steps,
        output_dir=save_dir,
        load_best_model_at_end=False,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="tensorboard", # if use_wandb else "none",
        metric_for_best_model="loss",
        save_total_limit=save_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=2,
    )

    tokenizer.save_pretrained(save_dir)
    
    logging.info("preprocessing datasets")
    random.shuffle(train_examples)
    if model.config.is_encoder_decoder:
        train_set = [extract_features_encoder_decoder(tokenizer, docstr, signature, code, max_len) for (docstr, signature, code, prefix, body) in train_examples]
        valid_set = [extract_features_encoder_decoder(tokenizer, docstr, signature, code, max_len) for (docstr, signature, code, prefix, body) in valid_examples]
    else:
        train_set = [extract_features_decoder_only(tokenizer, prefix, body, max_len) for (docstr, signature, code, prefix, body) in train_examples]
        valid_set = [extract_features_decoder_only(tokenizer, prefix, body, max_len) for (docstr, signature, code, prefix, body) in valid_examples]

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )


    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=valid_set,
        args=training_args,
        # data_collator=DataCollator(tokenizer, pad_to_multiple_of=8),
        data_collator=data_collator
    )

    if isinstance(model, PeftModelForCausalLM):
        model.config.use_cache = False
    #     old_state_dict = model.state_dict
    #     model.state_dict = (lambda _self, *_, **__: get_peft_model_state_dict(_self, old_state_dict())).__get__(
    #         model, type(model)
    #     )
    #     if torch.__version__ >= "2" and sys.platform != "win32":
    #         logging.info("compiling the model")
    #         model = torch.compile(model)
    trainer.train()