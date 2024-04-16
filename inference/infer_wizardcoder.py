import os
import sys
import fire
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils import chunks, generate_prompt, checkpoint, read_json_file, save_solution

MAX_INPUT_LEN = 4096
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sort_by_pcode_length(dataset):
    return sorted(dataset, key=lambda x: len(x["pcode"]))

def parse_model_output(output, prompt):
    if "### Response:" in output:
        final_output = output.split("### Response:")[1].strip()
    else:
        final_output = "[!] not found \"### Response:\""
        print("[!] not found \"### Response:\"")
    return final_output
    
def evaluate(
        prompts,
        tokenizer,
        model,
        max_length=4096,
        max_new_tokens=48
):
    inputs = tokenizer(prompts, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        top_k=10,
        top_p=0.95,
        temperature=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    outputs = [parse_model_output(output, prompt) for output, prompt in zip(decoded_outputs, prompts)]
    return decoded_outputs, outputs

def get_code(pcode, tokenizer: AutoTokenizer):
    tokenized_pcode = tokenizer(pcode,return_tensors='pt', max_length = MAX_INPUT_LEN-256, truncation=True).input_ids
    if tokenized_pcode.shape[1] == MAX_INPUT_LEN - 256: 
        pcode = tokenizer.batch_decode(tokenized_pcode)[0] + "......"
    return pcode

def main(
    load_8bit: bool = False,
    batch_size: int = 1, 
    base_model_path: str = "/archive/LLMs/WizardCoder-15B-V1.0",
    input_data_path = "../dataset/eval_dataset.json",
):
    assert base_model_path, (
        "Please specify a --base_model, e.g. --base_model='bigcode/starcoder'"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        load_in_8bit=load_8bit,
    )

    if not load_8bit:
        model.bfloat16()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    model_name = base_model_path.split('/')[-1]
    output_path = model_name + "_func_name.json"
    if os.path.exists(output_path):
        dataset, breakpoint = checkpoint(output_path)
    else:
        dataset = read_json_file(input_data_path)  
        dataset = sort_by_pcode_length(dataset)
        breakpoint = 0
    print("Total Num:", len(dataset), "Remain Num:", len(dataset[breakpoint:]))
    
    for chunk in tqdm(list(chunks(dataset[breakpoint:], batch_size))):
        prompt_text = [generate_prompt(get_code(sample["pcode"], tokenizer=tokenizer)) for sample in chunk]
        raw_outputs, outputs = evaluate(prompt_text, tokenizer, model)
        for sample, raw_output, output in zip(chunk, raw_outputs, outputs):
            sample['raw_output'] = raw_output
            sample['final_output'] = output

        save_solution(dataset, output_path)

if __name__ == "__main__":
    fire.Fire(main)