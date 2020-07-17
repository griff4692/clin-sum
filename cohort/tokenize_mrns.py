from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
import re
from time import time

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from constants import *
from utils import *

HTML_REGEX = r'(<[a-z][^>]+>|<\/?[a-z]>)'


def sent_segment(str):
    return [x.string.strip() for x in spacy_nlp(str).sents]


def tokenize_example(text):
    split_text = re.split(HTML_REGEX, text)
    bpe_final_text = []
    spacy_final_text = []
    sent_template = '<s> {} </s>'
    bpe_tok_ct = 0
    spacy_tok_ct = 0
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue
        if i > 0 and split_text[i - 1].strip() == '<h>':
            bpe_final_text.append(str)
            spacy_final_text.append(str)
        elif re.search(HTML_REGEX, str) is None:
            sents = sent_segment(str) if len(str) > 50 else [str]
            for sent in sents:
                bpe_tokens = tokenizer.tokenize(sent)
                bpe_tok_ct += len(bpe_tokens)
                bpe_token_str = ' '.join(bpe_tokens)
                bpe_sent_str = sent_template.format(bpe_token_str)
                bpe_final_text.append(bpe_sent_str)

                spacy_tokens = [x.text for x in spacy_nlp(sent)]
                spacy_tok_ct += len(spacy_tokens)
                spacy_token_str = ' '.join(spacy_tokens)
                spacy_sent_str = sent_template.format(spacy_token_str)
                spacy_final_text.append(spacy_sent_str)
        else:
            bpe_final_text.append(str)
            spacy_final_text.append(str)
    bpe_final_str = re.sub(r'\s+', ' ', ' '.join(bpe_final_text).strip())
    spacy_final_str = re.sub(r'\s+', ' ', ' '.join(spacy_final_text).strip())
    return bpe_tok_ct, bpe_final_str, spacy_tok_ct, spacy_final_str


def tokenize_mrn(mrn, mrn_counter=None, lock=None):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)

    examples_df.dropna(inplace=True)
    assert examples_df.shape[0] > 0

    spacy_source_toks, spacy_target_toks = [], []
    bpe_source_toks, bpe_target_toks = [], []
    spacy_source_n, spacy_target_n = [], []
    bpe_source_n, bpe_target_n = [], []
    for row in examples_df.to_dict('records'):
        bsn, bst, ssn, sst = tokenize_example(row['source_str'])
        btn, btt, stn, stt = tokenize_example(row['target_str'])

        bpe_source_toks.append(bst)
        bpe_target_toks.append(btt)
        bpe_source_n.append(bsn)
        bpe_target_n.append(btn)

        spacy_source_toks.append(sst)
        spacy_target_toks.append(stt)
        spacy_source_n.append(ssn)
        spacy_target_n.append(stn)

    examples_df['spacy_source_toks'] = spacy_source_toks
    examples_df['spacy_target_toks'] = spacy_target_toks
    examples_df['spacy_source_tok_ct'] = spacy_source_n
    examples_df['spacy_target_tok_ct'] = spacy_target_n
    examples_df['bpe_source_toks'] = bpe_source_toks
    examples_df['bpe_target_toks'] = bpe_target_toks
    examples_df['bpe_source_tok_ct'] = bpe_source_n
    examples_df['bpe_target_tok_ct'] = bpe_target_n
    examples_df.to_csv(examples_fn, index=False)

    with lock:
        mrn_counter.value += 1
        if mrn_counter.value % 1000 == 0:
            print('Processed {} MRNs'.format(mrn_counter.value))


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    with Manager() as manager:
        pool = Pool()  # By default pool will size depending on cores available
        mrn_counter = manager.Value('i', 0)
        lock = manager.Lock()
        pool.map(
            partial(
                tokenize_mrn, mrn_counter=mrn_counter, lock=lock
            ), mrns)
        pool.close()
        pool.join()

    duration(start_time)


