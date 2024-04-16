from __future__ import print_function, division
from iteration_utilities import deepflatten
from collections import Counter
import math
import re
import os
import hashlib
import sentencepiece as spm
from nltk.corpus import words
import nltk
nltk.download('words')

def is_required_func(func_name):
    if func_name.startswith('sub_') or func_name.startswith('_ZN') or func_name.startswith('_['):
        return False
    return True

# exclude consecutive upper letters
def split_normal_func_name(func_name):
    # remove all nonalpha char
    func_name = ''.join(list(map(lambda x: x if x.isalpha() else '_', func_name)))

    name_chars = [c for c in func_name]
    UpperCharIndex = [i for i, c in enumerate(name_chars) if c.isupper()]
    # print(UpperCharIndex)
    for i, idx in enumerate(UpperCharIndex):
        name_chars.insert(idx + i, '_')
    temp_name = ''.join(name_chars)
    tokens = list(filter(lambda x: len(x) != 0, temp_name.split('_')))
    tokens = list(map(lambda x: x.lower(), tokens))
    return tokens

def split_func_name(func_name):
    # remove all nonalpha char
    func_name = ''.join(list(map(lambda x: x if x.isalpha() else '_', func_name)))
    
    name_chars = [c for c in func_name]
    UpperCharIndex = [i for i, c in enumerate(name_chars) if c.isupper()]

    # [6, 7, 8, 15, 22, 23, 24, 25] to [6, 15, 22]
    def get_range(nums):
        left = 0
        right = 0

        result = []
        result.append(0)
        while right < len(nums):
            while right + 1 < len(nums) and nums[right] + 1 == nums[right + 1]:
                right += 1
            temp = []
            if right > left:
                if nums[left] not in result:
                    temp.append(nums[left])
                temp.append(nums[right]+1)
            elif right == left:
                pass
            result.extend(temp)
            left = right + 1
            right = left
        return result

    UpperCharRange = get_range(UpperCharIndex)
    if len(func_name) not in UpperCharRange:
        UpperCharRange.append(len(func_name))

    tmp_tokens = []
    for i in range(len(UpperCharRange)):
        if i != len(UpperCharRange)-1:
            tmp_tokens.append(func_name[UpperCharRange[i]:UpperCharRange[i+1]])

    tokens = []
    for t in tmp_tokens:
        if t.isupper():
            tokens.append(t.lower().replace('_', ''))
        else:
            tokens.extend(split_normal_func_name(t))
    return tokens

def is_cplusplus_func(func_name):
    return '(' in func_name and ')' in func_name

def match_cplusplus_func_name(full_name):
    method = full_name
    while method[-1] != ')':
        if method[-1] != ':':
            method = method[:-1]
        else:
            return ''

    regex_remove_type = re.compile(r'<[^<>]*>')
    while '<' in method and '>' in method:
        remove_list = regex_remove_type.findall(method)
        for t in remove_list:
            method = method.replace(t, '')

    regex_remove_args = re.compile(r'\([^\(\)]*\)')
    while '(' in method and ')' in method:
        remove_list = regex_remove_args.findall(method)
        for t in remove_list:
            method = method.replace(t, '')

    regex_remove_commet = re.compile(r'\[[^\[\]]*\]')
    while '[' in method and ']' in method:
        remove_list = regex_remove_commet.findall(method)
        for t in remove_list:
            method = method.replace(t, '')

    regex_remove_commet2 = re.compile(r'`[^`\']*\'')
    while '`' in method and '\'' in method:
        remove_list = regex_remove_commet2.findall(method)
        for t in remove_list:
            method = method.replace(t, '')

    if '::' in method:
        method = method.split('::')[-1]
        if ' ' in method:
            method = method.split(' ')[0]
    else:
        method = method.split(' ')[-1]

    if len([c for c in method if c.isalpha()]) == 0:
        return ''

    while method[-1] == '*':
        method = method[:-1]

    if method[0] == '~':
        method = method[1:]

    return method

def remove_items_from_list(items, l):
    return list(filter(lambda x: x not in items, l))

def split_sentence_piece(sp, tokens):

    tokenized = list(map(lambda x: sp.EncodeAsPieces(x), tokens))
    tokenized = list(deepflatten(tokenized, depth=1))

    tokenized = list(map(lambda x: x.replace('▁', ''), tokenized))

    tokenized = list(filter(lambda x: len(x) != 0, tokenized))
    return tokenized

def merge_suffix(tokens, vocab):
    suffixes = ['er', 'or', 'ed', 'ied', 'ing', 'ling', 'est', 's', 'es', 'ies', 'ify', 'ified', 'ly', 'ible', 'able', 'ize',
                'al', 'tion', 'ition', 'ous', 'ance', 'ence', 'ful', 'ment', 'hood']
    conditional_suffixes = [['t', 'ion'], ['e', 'd'], ['e', 'r']]

    if len(tokens) < 2:
        return tokens

    if tokens[-1] in suffixes:
        tokens[-2] = tokens[-2]+tokens[-1]
        tokens = tokens[:-1]
        return tokens
    for p1, p2 in conditional_suffixes:
        if tokens[-2].endswith(p1) and tokens[-1].endswith(p2):
            tokens[-2] = tokens[-2] + tokens[-1]
            tokens = tokens[:-1]
            return tokens
    if tokens[-2] in vocab and tokens[-1] not in vocab and tokens[-2]+tokens[-1] in vocab:
        tokens = tokens[-2]+tokens[-1]
    return tokens

def get_files_with_specific_ext_from_dir(rootname, exts):
    if isinstance(exts, str):
        exts = [exts]

    results = []
    for main_dir, _, file_name_list in os.walk(rootname):
        for f in file_name_list:
            file_path = os.path.join(main_dir, f)
            for ext in exts:
                if file_path.endswith(ext):
                    results.append(file_path)
                    
    results = list(set(results))
    return results

def md5(string):
    m = hashlib.md5()
    m.update(bytes(string, encoding='utf-8'))
    return m.hexdigest()

def shannon_entropy(string):
    count = Counter(i for i in string).most_common()
    f_len = len(string)
    entropy = -sum(j/f_len*(math.log(j/f_len)) for i,j in count) #shannon entropy
    return entropy

def unsigned2signed(number, width):
    if number > 2**(width-1) - 1:
        number = 2 ** width-number
        number = 0-number
    return hex(number)

sp = spm.SentencePieceProcessor()
sp.load(os.path.join('model', 'sentencepiece.model'))
vocab = words.words()

def my_split_func_name(name):
    return merge_suffix(split_sentence_piece(sp, split_func_name(name)), vocab)