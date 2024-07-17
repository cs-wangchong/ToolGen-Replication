from transformers import PreTrainedTokenizerBase

def build_prompt_decoder_only(prefix):
    return prefix

def extract_features_decoder_only(tokenizer:PreTrainedTokenizerBase, prefix, body, max_len):
    prompt = build_prompt_decoder_only(prefix)
    input_text = f"{prompt}{body}"
    features = tokenizer(
        input_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )
    features["input_ids"].append(tokenizer.eos_token_id)
    features["attention_mask"].append(1)
    # prompt_len = len(tokenizer(
    #     prompt,
    #     truncation=True,
    #     max_length=max_len,
    #     padding=False,
    #     return_tensors=None,
    # )["input_ids"])
    # features["labels"] = [-100] * prompt_len + features["input_ids"][prompt_len:]
    features["labels"] = features["input_ids"]
    return features


def build_prompt_encoder_decoder(docstr, signature):
    return f"### Description:\n{docstr.strip()}\n\n### Signature:\n{signature.strip()}"

def extract_features_encoder_decoder(tokenizer:PreTrainedTokenizerBase, docstr, signature, code, max_len):
    input_text = build_prompt_encoder_decoder(docstr, signature)
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
