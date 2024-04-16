from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
import json
import evaluate
import sacrebleu
from evaluate.utils.file_utils import DownloadConfig
from rouge_score import rouge_scorer, scoring

data_path = "../dataset/CodeLlama-7b-Instruct-hf_comment.json"
print(data_path)
with open(data_path, "r") as f:
    data = json.load(f)

references=[]
candidates=[]
for item in data:
    references.append(item["chatgpt_raw"])
    candidates.append(item["final_output"])

smooth_bleu_metric = []
meteor_metric = evaluate.load('meteor')
rouge_metric = evaluate.load('rouge')

bleu = sacrebleu.corpus_bleu(candidates, [references], smooth_method='exp')
meteor_score = meteor_metric.compute(predictions=candidates, references=references)['meteor']

rouge_score = rouge_metric.compute(predictions=candidates, references=references)['rougeL']

print("Smooth BLEU Score:", bleu)
print(f"Average METEOR: {meteor_score*100:.2f}, Average Rouge-L: {rouge_score*100:.2f}")