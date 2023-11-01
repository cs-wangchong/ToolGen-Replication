import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoTokenizer, AddedToken, RobertaTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM, T5ForConditionalGeneration

from coder.constants import *

torch.manual_seed(42)  # pytorch random seed


def init_codellama(
    model_name="codellama/CodeLlama-7b-Python-hf",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
        embs = model.resize_token_embeddings(len(tokenizer))
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # modules_to_save=["embed_tokens"]
        )
        model.enable_input_require_grads()
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.to(device)

    def build_prompt(docstr, signature):
        return f"### Description:\n{docstr.strip()}\n\n### Signature:\n{signature.strip()}\n\n### Code:\n"
        # return f"### Function Description: \n{docstr.strip()}\n\n### Function Signature: \n{signature.strip()}\n\n### Function Code:\n"
        # return f"### Description: {docstr.strip()}\n### Signature: {signature.strip()}\n### Code:\n"
    
    def extract_features(docstr, signature, code, max_len):
        input_text = f"{build_prompt(docstr, signature)}{code}"
        features = tokenizer(
            input_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        features["labels"] = features["input_ids"].copy()
        return features
    
    return model, tokenizer, build_prompt, extract_features


def init_codet5(
    model_name="Salesforce/codet5p-220m-py",
    checkpoint=None,
    additional_tokens=[PLM_LSP_POINT],
    device="cuda"
):
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=False)
    
    if checkpoint is None:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    additional_tokens = [] if additional_tokens is None else additional_tokens
    if len(additional_tokens) > 0:
        tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in additional_tokens])
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)

    def build_prompt(docstr, signature):
        return f"### Function Description: \n{docstr.strip()}\n\n### Function Signature: \n{signature.strip()}"
        # return f"Description: {docstr.strip()}\nSignature: {signature.strip()}"

    def extract_features(docstr, signature, code, max_len):
        input_text = f"{build_prompt(docstr, signature)}"
        features = tokenizer(
            input_text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        output_features = tokenizer(
            code,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
        features["labels"] = output_features["input_ids"]
        return features
    
    return model, tokenizer, build_prompt, extract_features