# ToolGen-Replication

## Datasets
Download the augmented dataset and benchmark dataset from [ToolGen-Datasets](https://drive.google.com/drive/folders/1AAB94jLarnO8INJMOY4aBPhYjOzQQLhr?usp=sharing) and put them into a folder named `data/datasets`.


## Fine-tuning
Run the script in the folder `script/train_model.py` to fine-tune the corresponding code LLMs, including CodeGPT, CodeT5, and CodeLlama. The fine-tuned models are stored in the folder `models/`.

Arguments for executing the trianing script:
- `--model`: the name of base model, including `codegpt`, `codet5`, and `codellama`
- `--lsp`: using augmented training set or original training set, lsp represents "Language Server Protocol"
- `--lr`: learning rate
- `--epoch`: training epoch numners
- `--max_len`: max token numbers of the code
- `--train_batch`: training batch size
- `--valid_batch`: validation batch size

An execution example: 
```shell
CUDA_VISIBLE_DEVICES=0 python script/train_model.py --model=codet5 --lr=5e-6 --epoch=10 --max_len=256 --train_batch=64 --valid_batch=64 --lsp
```

We have released some fine-tuned checkpoints at [ToolGen-Checkpoints](https://drive.google.com/drive/folders/1VRxceCkrazjyY4bgiVC3Esm-A-DhgWDU?usp=sharing). In the folder, there are the following checkpoints:
- CodeGPTs:
    - `CodeGPT-small-py/nolsp-xxxx`: CodeGPT model fine-tuned on **original** functions
    - `CodeGPT-small-py/lsp-xxxx`: CodeGPT model fine-tuned on **augmented** functions
- CodeT5s:
    - `codet5p-220m-py/nolsp-xxxx`: CodeT5 model fine-tuned on **original** functions
    - `codet5p-220m-py/lsp-xxxx`: CodeT5 model fine-tuned on **augmented** functions
- CodeLlamas:
    - `CodeLlama-7b-Python-hf/nolsp-xxxx`: CodeLlama model fine-tuned on **original** functions
    - `CodeLlama-7b-Python-hf/lsp-xxxx`: CodeLlama model fine-tuned on **augmented** functions

`xxxx` is the timestamp.

## Evaluation
Run `script/eval_coder.py` to evaluate the fine-tuned code LLMs in repo-level code generation. 


Arguments for executing the evaluation script:
- `--dir`: the folder pointing to the fine-tuned model
- `--max_len`: max token numbers for the generated code
- `--batch`: batch size

An example:
```shell
CUDA_VISIBLE_DEVICES=0 python script/eval_coder.py --dir=models/codet5p-220m-py/lsp-240701-111433 --batch=64
```

Note that, to calculate dependency-based metrics, code repositories should be downloaded from [ToolGen-TestRepos](https://drive.google.com/file/d/1O1sEn48m3P3qyqFpVH9DYKg7saO8Roqa/view?usp=sharing). 
