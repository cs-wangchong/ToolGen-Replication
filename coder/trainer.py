import logging
import math
from pathlib import Path
import random
import sys
from typing import Union, Callable

import torch
from peft import (
    PeftModelForCausalLM,
    get_peft_model_state_dict,
)
from transformers import PreTrainedTokenizer
from transformers import PreTrainedModel
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

from coder.constants import *

torch.manual_seed(42)  # pytorch random seed

def train(
        model:PreTrainedModel,
        tokenizer:PreTrainedTokenizer,
        extract_features:Callable,
        train_examples,
        valid_examples,
        epoch_num=2,
        learning_rate=1e-4,
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
        max_steps=max_steps,
        learning_rate=learning_rate,
        # fp16=True,
        logging_steps=logging_steps,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=valid_steps,
        save_steps=valid_steps,
        output_dir=save_dir,
        load_best_model_at_end=False,
        # group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="tensorboard", # if use_wandb else "none",
        metric_for_best_model="loss",
        save_total_limit=save_limit
    )
    
    logging.info("preprocessing datasets")
    random.shuffle(train_examples)
    train_set = [extract_features(docstr, signature, code, max_len) for (docstr, signature, code) in train_examples]
    valid_set = [extract_features(docstr, signature, code, max_len) for (docstr, signature, code) in valid_examples]

    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=valid_set,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # if isinstance(model, PeftModelForCausalLM):
    #     model.config.use_cache = False
    #     old_state_dict = model.state_dict
    #     model.state_dict = (lambda _self, *_, **__: get_peft_model_state_dict(_self, old_state_dict())).__get__(
    #         model, type(model)
    #     )
    #     if torch.__version__ >= "2" and sys.platform != "win32":
    #         logging.info("compiling the model")
    #         model = torch.compile(model)
    trainer.train()