import os
import json
import pickle
from utils import my_split_func_name
# import nltk
# nltk.download('words')

load_summarization = True
summarization_path = './summarization.txt'
summarization_pickle_path = summarization_path+".pkl"

if load_summarization:
    if os.path.exists(summarization_pickle_path):
        # load from pkl
        with open(summarization_pickle_path, "rb") as f:
           summarization_list = pickle.load(f)
    else:
        with open(summarization_path, 'r', encoding='utf-8') as f:
            content = f.read().split('\n')
        summarization_list = list(map(lambda x: x.split(','), content))
        # save to pkl
        with open(summarization_pickle_path, "wb") as f:
            pickle.dump(summarization_list,f)


def get_aprf(preds:list, refs:list):
    assert len(preds) == len(refs)
    accs = []
    precisions = []
    recalls = []
    f1s = []
    for p, r in zip(preds, refs):

        p_tokens = p.split(' ')
        r_tokens = r.split(' ')

        r_tokens = list(filter(lambda x: x!="", r_tokens))
        if len(r_tokens) == 0:
            continue

        if load_summarization:
            r_tokens_extend = []
            for r_t in r_tokens:
                r_tokens_extend.append(r_t)
                for summ_l in summarization_list:
                    if r_t in summ_l:
                        r_tokens_extend.extend(summ_l)
            r_tokens_extend = list(set(r_tokens_extend))

            acc = len([x for x in p_tokens if x in r_tokens_extend]) / len(p_tokens)
            precision = sum([1 if p_t in r_tokens_extend else 0 for p_t in p_tokens]) / len(p_tokens)
            recall = sum([1 if p_t in r_tokens_extend else 0 for p_t in p_tokens]) / len(r_tokens)

        else:
            acc = len([x for x in p_tokens if x in r_tokens]) / len(p_tokens)
            precision = sum([1 if p_t in r_tokens else 0 for p_t in p_tokens]) / len(p_tokens)
            recall = sum([1 if p_t in r_tokens else 0 for p_t in p_tokens]) / len(r_tokens)
            # precision = sum([1 if p_t in r_tokens else 0 for p_t in p_tokens]) / (sum([1 if p_t in r_tokens else 0 for p_t in p_tokens]) + sum([1 if p_t not in r_tokens else 0 for p_t in p_tokens]))
            # recall = sum([1 if p_t in r_tokens else 0 for p_t in p_tokens]) / (sum([1 if p_t in r_tokens else 0 for p_t in p_tokens]) + sum([1 if r_t not in p_tokens else 0 for r_t in r_tokens]))

        f1 = 2*precision*recall / (precision+recall+1e-9)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_accuracy = sum(accs) / len(accs)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)

    return {'accuracy': avg_accuracy,'precision':avg_precision, 'recall':avg_recall, 'f1':avg_f1}

def print_result_from_json(result_json):
    with open(result_json,"r",encoding="utf-8") as f:
        data = json.load(f)
    
    print("[-] all item: ",len(data))

    print("[-] splitting function name...")
    preds = [item[1] for item in data]
    refs = [item[0] for item in data]

    refs = [" ".join(my_split_func_name(name)) for name in refs]
    preds = [" ".join(my_split_func_name(name)) for name in preds]

    metrics = get_aprf(preds, refs)
    print(json.dumps(metrics))

if __name__ == '__main__':
    filepath = '../dataset/CodeLlama-7b-Instruct-hf_func_name.json'
    print(filepath.split('/')[-1])
    print_result_from_json(filepath)