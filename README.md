# ToolGen-Replication

## Datasets
Download the augmented dataset and benchmark dataset from google drive [ToolGen-Data](https://drive.google.com/drive/folders/16ddTvkknl9eDnX1J6tDJLxc-04hEetDK?usp=drive_link) and put them into a folder named `data/datasets`.


## Fine-tuning
Run the scripts in the folder `script/train` to fine-tune the corresponding code LLMs, including CodeGPT, CodeT5, and CodeLlama. The fine-tuned models are stored in the folder `models-lsp/`.

## Evaluation
Run `script/eval_coder.py` to evaluate the fine-tuned code LLMs in repo-level code generation. 

Use `-model {model_name}` and `-version {model_path}` to specify the name of base model (i.e., `codegpt`, `codet5`, and `codellama`) and the fine-tuned model path (e.g., `models-lsp/CodeLlama-7b-Python-hf-231108-114340`), respectively.


