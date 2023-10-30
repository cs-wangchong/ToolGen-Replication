from transformers import T5Tokenizer, RobertaTokenizer, T5ForConditionalGeneration, AddedToken
import re

def clean_lsp(code):
    code = code.replace(' <lsp> ', '')
    return code


tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer.add_tokens([AddedToken("<lsp>", rstrip=False, lstrip=False, single_word=False)])

# model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

code = '''def get_tables_with_type(self, ip_type):
        tables = []
        <lsp>for key, d <lsp>in <lsp>self.<lsp>metadata.<lsp>items():
            <lsp>if <lsp>d['type'] == <lsp>ip_type:
                tables.<lsp>append(<lsp>key)
        <lsp>return <lsp>tables'''
print(tokenizer.tokenize(code))
input_ids = tokenizer(code, return_tensors="pt").input_ids
output = tokenizer.batch_decode(input_ids, False, False)[0]
print(output)
print(clean_lsp(output))

# # simply generate one code span
# generated_ids = model.generate(input_ids, max_length=150)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# # this prints "{user.username}"


