# BinaryLLMs-Eval

## How Far Have We Gone in Binary Code Understanding Using Large Language Models

## 1. Environment Setup

`pip install -r requirements.txt`

## 2. Inference

We provide here scripts to infer locally deployed LLMs and call ChatGPT via API.

`CUDA_VISIBLE_DEVICES=0 python infer_llama.py`

The evaluation data is in the **dataset** folder, and the specific prompts are provided in the **utils.py** file.

## 3. Evaluation

Calculate the **Precision, Recall, and F1-score** metrics of **function name recovery** task

`python cal_funcname_metrics.py`

Calculate the **BLEU-4, METEOR, and Rouge-L** metrics of **binary code summarization** task

`python cal_summarization_metrics.py`