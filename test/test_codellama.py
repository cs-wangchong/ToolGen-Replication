import torch
from transformers import AutoModel, AutoTokenizer, pipeline, CodeLlamaTokenizerFast, AddedToken
from transformers.models.llama.modeling_llama import LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf", use_fast=False)
# model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf", torch_dtype=torch.float16, device_map="cuda",)
tokenizer.add_tokens([AddedToken(t, rstrip=False, lstrip=False) for t in ["<lsp>"]])

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

code = '''def get_tables_with_type(self, ip_type):
        tables = []
        <lsp>for key, d <lsp>in <lsp>self.<lsp>metadata.<lsp>items():
            <lsp>if <lsp>d['type'] == <lsp>ip_type:
                tables.<lsp>append(<lsp>key)
        <lsp>return <lsp>tables'''
print(tokenizer.tokenize(code))

# print(tokenizer._convert_id_to_token(2))


# inputs = ["Description: Report usage of training parameters.\nFunction Signature: report(self)\n\nCode:\n"]



# input_ids = tokenizer(inputs, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt").input_ids
# input_ids = input_ids.to("cuda")
# attention_mask = input_ids.ne(tokenizer.pad_token_id)

# output = model.generate(
#     input_ids,
#     attention_mask=attention_mask,
#     max_length=128,
#     repetition_penalty=2.5,
#     num_beams=1,
#     num_return_sequences=1,
# )
# print(tokenizer.batch_decode(output, skip_special_tokens=True))