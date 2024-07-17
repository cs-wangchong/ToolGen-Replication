import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AddedToken, RobertaTokenizer
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, T5ForConditionalGeneration, GPT2LMHeadModel, CodeGenModel

from coder.constants import *


torch.manual_seed(42)  # pytorch random seed


def init_deepseek_lora(
    model_name="deepseek-ai/deepseek-coder-1.3b-base",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
        model.resize_token_embeddings(len(tokenizer))
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.to(device)
    
    return model, tokenizer


def init_deepseek(
    model_name="deepseek-ai/deepseek-coder-1.3b-base",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
    if checkpoint is None:
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    model.to(device)

    return model, tokenizer


def init_codellama_lora(
    model_name="codellama/CodeLlama-7b-Python-hf",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # ),
        # load_in_8bit=True,
        device_map=("auto" if device == "cuda" else device),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
        model.resize_token_embeddings(len(tokenizer))
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.to(device)
    
    return model, tokenizer


def init_codellama(
    model_name="codellama/CodeLlama-7b-Python-hf",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
    if checkpoint is None:
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            device_map=device,
        )
    
    model.to(device)
    
    return model, tokenizer


def init_codegen(
    model_name="Salesforce/codegen-350M-mono",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
    if checkpoint is None:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    
    model.to(device)
    
    return model, tokenizer



def init_codegpt(
    model_name="microsoft/CodeGPT-small-py",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"    
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
    if checkpoint is None:
        model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(checkpoint)
    
    model.to(device)
    
    return model, tokenizer



def init_codet5(
    model_name="Salesforce/codet5p-220m-py",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=False)
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
    if checkpoint is None:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
   
    model.to(device)

    return model, tokenizer
