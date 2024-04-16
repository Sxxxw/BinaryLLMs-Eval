import re
import time
import json
import jsonlines

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_jsonline_file(file_path):
    with jsonlines.open(file_path) as reader:
        data = [d for d in reader]
    return data

def _write_file(dataset, path):
    if path.endswith('.jsonl'):
        with jsonlines.open(path, 'w') as writer:
            for data in dataset:
                writer.write(data)
    elif path.endswith('.json'):
        with open(path, 'w') as writer:
            json.dump(dataset, writer, indent=4)
    else:
        raise NotImplementedError("[!] Unsupported file format. Please use .json or .jsonl file.")

def save_solution(dataset: list, path: str):
    return _write_file(dataset, path)

def checkpoint(save_file_path):
    if save_file_path.endswith('.json'):
        done_data = read_json_file(save_file_path)
    elif save_file_path.endswith('.jsonl'):
        done_data = read_jsonline_file(save_file_path)
    else:
        raise NotImplementedError
    
    finish_num = 0
    for item in done_data:
        if "final_output" in item:
            finish_num += 1
        else:
            break
    print(f'[+] checkpoint: {finish_num}/{len(done_data)} data has been done.')
    time.sleep(3)
    return done_data, finish_num

def find_whole_word(word, text):
    pattern = r'\b{}\b'.format(re.escape(word))
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        return None

def parse_model_output(output, prompt):
    if output.startswith(prompt):
        output = output[len(prompt):].strip()
    pattern = r'```[a-zA-Z]+\n(.+?)\n```'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

PROMPT_FUNCNAME_ZEROSHOT = """
### Instruction:
Please imagine you are an experienced binary reverse engineer. The following is a stripped decompiled C function, your task is to understand it and response with a descriptive function name to replace <FUNCTION>. Your response format is 'The descriptive function name is xxx', do not explain.\n

```C
{pseudo_code}
```

### Response:
"""

PROMPT_FUNCNAME_FEWSHOT_EXAMPLE = """
Please imagine you are an experienced binary reverse engineer. The following is a stripped decompiled C function, your task is to understand it and response with a descriptive function name to replace <FUNCTION>. Your response format is 'The descriptive function name is xxx', do not explain.\n

Here are two examples:

Example 1 decompiled C function:
```C
void __fastcall <FUNCTION>(const char *a1, __int64 a2)\n{\n  int *v2; // rax\n  char *v3; // rax\n\n  if ( !a2 )\n    __assert_fail(\"dictFileStat != NULL\", \"fileio.c\", 0x2ADu, \"FIO_getDictFileStat\");\n  if ( a1 )\n  {\n    if ( !(unsigned int)sub_1AEC79(a1, a2) )\n    {\n      if ( dword_1E70F8 > 0 )\n        fwrite(\"zstd: \", 1uLL, 6uLL, stderr);\n      if ( dword_1E70F8 > 4 )\n        fprintf(stderr, \"Error defined at %s, line %i : \\n\", \"fileio.c\", 689LL);\n      if ( dword_1E70F8 > 0 )\n        fprintf(stderr, \"error %i : \", 31LL);\n      if ( dword_1E70F8 > 0 )\n      {\n        v2 = __errno_location();\n        v3 = strerror(*v2);\n        fprintf(stderr, \"Stat failed on dictionary file %s: %s\", a1, v3);\n      }\n      if ( dword_1E70F8 > 0 )\n        fwrite(\" \\n\", 1uLL, 2uLL, stderr);\n      exit(31);\n    }\n    if ( !(unsigned int)sub_1AEDE8(a2) )\n    {\n      if ( dword_1E70F8 > 0 )\n        fwrite(\"zstd: \", 1uLL, 6uLL, stderr);\n      if ( dword_1E70F8 > 4 )\n        fprintf(stderr, \"Error defined at %s, line %i : \\n\", \"fileio.c\", 693LL);\n      if ( dword_1E70F8 > 0 )\n        fprintf(stderr, \"error %i : \", 32LL);\n      if ( dword_1E70F8 > 0 )\n        fprintf(stderr, \"Dictionary %s must be a regular file.\", a1);\n      if ( dword_1E70F8 > 0 )\n        fwrite(\" \\n\", 1uLL, 2uLL, stderr);\n      exit(32);\n    }\n  }\n}\n
```

Example 1 response:
The descriptive function name is `FIO_getDictFileStat`

Example 2 decompiled C function:
```C
unsigned __int64 __fastcall <FUNCTION>(_QWORD *a1, __int64 a2)\n{\n  _QWORD *v2; // rax\n  _QWORD *v3; // rax\n  unsigned __int64 v5; // [rsp+A8h] [rbp-8h]\n\n  v5 = __readfsqword(0x28u);\n  if ( a1 )\n    v2 = a1 + 338;\n  else\n    v2 = 0LL;\n  sub_5E3AFC(v2, a2);\n  if ( sub_5E4367(a2) )\n  {\n    je_malloc_mutex_lock_slow(a2);\n    *(_BYTE *)(a2 + 104) = 1;\n    _mm_mfence();\n  }\n  sub_5E438A(a1, (_QWORD *)a2);\n  if ( a1 )\n    v3 = a1 + 338;\n  else\n    v3 = 0LL;\n  sub_5E3D72(v3, a2);\n  return __readfsqword(0x28u) ^ v5;\n}\n
```

Example 2 response:
The descriptive function name is `malloc_mutex_lock`

"""

PROMPT_FUNCNAME_FEWSHOT = """
Input decompiled C function:
```C
{pseudo_code}
```

Response:
"""

PROMPT_COMMENT_ZEROSHOT = """
### Instruction:
Please imagine you are an experienced binary reverse engineer. The following is a stripped decompiled C function, your task is to understand it and generate a short comment to the function describing its functionality. No more than 96 words.

```C
{pseudo_code}
```

### Response:
"""

PROMPT_COMMENT_FEWSHOT_EXAMPLE = """
Please imagine you are an experienced binary reverse engineer. The following is a stripped decompiled C function, your task is to understand it and generate a short comment to the function describing its functionality. No more than 96 words.

Here are two examples:

Example 1 decompiled C function:
```C
void __fastcall <FUNCTION>(const char *a1, __int64 a2)\n{\n  int *v2; // rax\n  char *v3; // rax\n\n  if ( !a2 )\n    __assert_fail(\"dictFileStat != NULL\", \"fileio.c\", 0x2ADu, \"FIO_getDictFileStat\");\n  if ( a1 )\n  {\n    if ( !(unsigned int)sub_1AEC79(a1, a2) )\n    {\n      if ( dword_1E70F8 > 0 )\n        fwrite(\"zstd: \", 1uLL, 6uLL, stderr);\n      if ( dword_1E70F8 > 4 )\n        fprintf(stderr, \"Error defined at %s, line %i : \\n\", \"fileio.c\", 689LL);\n      if ( dword_1E70F8 > 0 )\n        fprintf(stderr, \"error %i : \", 31LL);\n      if ( dword_1E70F8 > 0 )\n      {\n        v2 = __errno_location();\n        v3 = strerror(*v2);\n        fprintf(stderr, \"Stat failed on dictionary file %s: %s\", a1, v3);\n      }\n      if ( dword_1E70F8 > 0 )\n        fwrite(\" \\n\", 1uLL, 2uLL, stderr);\n      exit(31);\n    }\n    if ( !(unsigned int)sub_1AEDE8(a2) )\n    {\n      if ( dword_1E70F8 > 0 )\n        fwrite(\"zstd: \", 1uLL, 6uLL, stderr);\n      if ( dword_1E70F8 > 4 )\n        fprintf(stderr, \"Error defined at %s, line %i : \\n\", \"fileio.c\", 693LL);\n      if ( dword_1E70F8 > 0 )\n        fprintf(stderr, \"error %i : \", 32LL);\n      if ( dword_1E70F8 > 0 )\n        fprintf(stderr, \"Dictionary %s must be a regular file.\", a1);\n      if ( dword_1E70F8 > 0 )\n        fwrite(\" \\n\", 1uLL, 2uLL, stderr);\n      exit(32);\n    }\n  }\n}\n
```

Example 1 response:
This function, `FIO_getDictFileStat`, takes in a file name and a `stat_t` structure. It first checks if the `dictFileStat` structure is not NULL, and if the `fileName` is NULL, it directly returns. \n\nNext, it uses the `UTIL_stat` function to get the statistics of the file specified by `fileName` and stores it in the `dictFileStat` structure. If the `UTIL_stat` function fails, it throws an exception with error code 31 and a specific error message.\n\nThen, it checks if the file specified by `fileName` is a regular file using the `UTIL_isRegularFileStat` function. If it is not a regular file, it throws an exception with error code 32 and a specific error message.

Example 2 decompiled C function:
```C
__int64 __fastcall <FUNCTION>(__int64 a1, _DWORD *a2)\n{\n  __int64 v2; // r9\n  __int64 result; // rax\n  int v4; // edx\n  __int64 v5; // r8\n  __int64 v6; // r9\n  int v7; // ecx\n  __int64 v8; // rdx\n\n  v2 = *(_QWORD *)(a1 + 48);\n  result = -1LL;\n  v4 = HIBYTE(*a2);\n  LODWORD(v5) = *a2 & 0xFFFFFF;\n  if ( v4 != 255 )\n  {\n    if ( v4 )\n    {\n      v8 = (unsigned int)(v4 - 1);\n      result = *(_QWORD *)(*(_QWORD *)(v2 + 16LL * (unsigned int)v5) + 8 * v8);\n    }\n    else\n    {\n      v5 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 1360)\n                               + 2\n                               * (*(unsigned __int8 *)(a1 + 1349)\n                                + (unsigned __int64)(unsigned int)((_DWORD)v5 << *(_DWORD *)(a1 + 1356))));\n      v6 = 16 * v5 + v2;\n      v7 = *(_DWORD *)(v6 + 8);\n      if ( !v7 )\n        return result;\n      LODWORD(v8) = v7 - 1;\n      result = *(_QWORD *)(*(_QWORD *)v6 + 8LL * (unsigned int)(v7 - 1));\n    }\n    *a2 = v5 | ((_DWORD)v8 << 24);\n  }\n  return result;\n}\n
```

Example 2 response:
The `smack_search_next_end` function is used to search for the next occurrence of a pattern's end in a given input string using the SMACK algorithm. It takes a SMACK data structure, `smack`, and the current state of the search, `current_state`, as inputs. \n\nThe function first checks if all patterns have been matched, indicated by `current_matches` being equal to 0xFF. If so, it returns `SMACK_NOT_FOUND`.\n\nIf patterns haven't been completely matched, the function retrieves the next matched pattern, updating `current_matches` and returning the corresponding pattern ID.\n\nIn case no match has been found yet, it searches for a match based on the anchor character at the end of patterns. If a match is found, it returns the ID of the first matched pattern.\n\nThe function updates `current_state` with the current row and number of matches and returns the found pattern ID.

"""

PROMPT_COMMENT_FEWSHOT = """
Input decompiled C function:
```C
{pseudo_code}
```

Response:
"""

template_dict = {
    "func_name_zeroshot": PROMPT_FUNCNAME_ZEROSHOT,
    "func_name_fewshot": PROMPT_FUNCNAME_FEWSHOT,
    "func_comment_zeroshot": PROMPT_COMMENT_ZEROSHOT,
    "func_comment_fewshot": PROMPT_COMMENT_FEWSHOT
} 

def generate_prompt(pseudo_code, template="func_name_zeroshot"):
    prompt_template = template_dict[template]
    if template=="func_name_fewshot":
        return PROMPT_FUNCNAME_FEWSHOT_EXAMPLE + prompt_template.format(pseudo_code=pseudo_code)
    if template=="func_comment_fewshot":
        return PROMPT_COMMENT_FEWSHOT_EXAMPLE + prompt_template.format(pseudo_code=pseudo_code)
    return prompt_template.format(pseudo_code=pseudo_code)