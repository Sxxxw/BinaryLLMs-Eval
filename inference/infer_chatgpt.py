'''
@desc: query chatgpt
@requirements: termcolor, openai, tiktoken
@usage: python3 thisfile.py
'''
import re
import os
import json
import time
import jsonlines
import random
from tqdm import tqdm
from openai import OpenAI
import tiktoken
from termcolor import colored
        
api_key_list = [
                ('sk-xxxxx', 'free')
                ]             
MODEL = "gpt-3.5-turbo-16k"
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
MAX_TOKEN_LENGTH = 16000

json_file_path = "../dataset/dataset_x64_o0.json"
match = re.search(r'_(\w+_\w+)', json_file_path)
save_file_path = "./gpt-3.5-turbo-16k_func_name_{}.jsonl".format(match.group(1))

PROMPT = """
### Instruction:
Please imagine you are an experienced binary reverse engineer. The following is a stripped decompiled C function, your task is to understand it and response with a descriptive function name to replace <FUNCTION>. Your response format is 'The descriptive function name is xxx', do not explain.\n

```C
{code}
```

### Response:
"""

random.seed(233)
TYPES_OF_LIMITS_PER_MINNUTE = {
    "free": 60/3,
    "tier1": 60/500,
    "tier2": 60/5000,
    "tier3": 60/5000,
    "tier4": 60/10000,
    "tier5": 60/10000,
}

class KeyPool():
    '''
    code example:
    key_list = [
        ('sk-xxxxx', 'free'),
    ]
    key_pool = KeyPool(key_list)
    while NEED_QUERY():
        key = key_pool.get_key():
        content = do_query()
        status = key_pool.judge_status(content)
        key_pool.feedback(key, status)
    '''
    def __init__(self, key_list) -> None:
        random.shuffle(key_list)
        self.key_list = []
        for item in key_list:
            _key,_type = item
            self.key_list.append({
                "key": _key,
                "type": _type,
                "ok_time": time.time(),
            })
        self.cur_idx = 0
        print(colored('[+] load keys:','blue'))
        print(json.dumps(self.key_list, indent=4))

    def get_key(self):
        while True:
            item = self.key_list[self.cur_idx]
            self.cur_idx = (self.cur_idx + 1) % len(self.key_list)
            if item['ok_time'] <= time.time():
                return item['key']
    
    def feedback(self, key, status):
        for item in self.key_list:
            if item['key'] == key:
                if status == 'good':
                    _type = item['type']
                    item['ok_time'] = time.time() + TYPES_OF_LIMITS_PER_MINNUTE[_type]
                elif status == 'PRM_limit':
                    _type = item['type']
                    item['ok_time'] = time.time() + TYPES_OF_LIMITS_PER_MINNUTE[_type]               
                elif status == 'RPD_limit':
                    _type = item['type']
                    item['ok_time'] = time.time() + 3600*24
                elif status == 'exceeded_quota':
                    _type = item['type']
                    item['ok_time'] = time.time() + 1e9
                else:
                    raise NotImplementedError
    
    def judge_status(self, content):
        if 'Error code: 429' in content:
            if 'RPM' in content:
                return "PRM_limit"
            elif 'RPD' in content:
                return "RPD_limit"
            elif 'You exceeded your current quota' in content:
                return "exceeded_quota"
            else: 
                raise NotImplementedError
        else:
            return "good"

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_jsonline_file(file_path):
    with jsonlines.open(file_path) as reader:
        data = []
        for d in reader:
            print(d)
            data.append(d)
    return data

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def write_jsonline_file(file_path, data):
    with jsonlines.open(file_path, 'a') as writer:
        writer.write(data)

def checkpoint(data, save_file_path):
    if not os.path.exists(save_file_path):
        return data
    if save_file_path.endswith('.json'):
        done_data = read_json_file(save_file_path)
    elif save_file_path.endswith('.jsonl'):
        done_data = read_jsonline_file(save_file_path)
    else:
        raise NotImplementedError

    done_fids = {d['id_num']:d for d in done_data}
    new_data = []
    cnt = 0
    for item in data:
        if item['id_num'] in done_fids:
            new_data.append(done_fids[item['id_num']])
            cnt+=1
        else:
            new_data.append(item)
    print('[+] checkpoint:', cnt, 'data has been done.')
    return new_data

def call_chatgpt(prompt, api_key):
    try:
        client = OpenAI(
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            messages = [
                {"role": "user", "content": prompt}
                ],
            model = MODEL,
        )
    except Exception as e:
        return str(e)
    return completion.choices[0].message.content

def parse_chatgpt_output(output):
    pass

def main(sleep_time=20):
    data = read_json_file(json_file_path)
    data = checkpoint(data, save_file_path)
    
    key_pool = KeyPool(api_key_list)

    for item in tqdm(data):
        if 'chatgpt_raw' in item:
            continue

        while True:
            api_key = key_pool.get_key()
            
            code = item['pcode']
            code = tokenizer.encode(code)
            if len(code) >= MAX_TOKEN_LENGTH:
                code = code[:MAX_TOKEN_LENGTH]
            code = tokenizer.decode(code)

            input_text = PROMPT.format(code=code)
            response = call_chatgpt(input_text, api_key)
            print(colored("[-] response:\n",'blue'),colored(response,'green'))

            status = key_pool.judge_status(response)
            key_pool.feedback(api_key, status)
            
            if status == 'good':
                break
            
        item['chatgpt_raw'] = response

        write_jsonline_file(save_file_path,item)

if __name__ == '__main__':
    main()