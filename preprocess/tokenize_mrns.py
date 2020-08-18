from collections import defaultdict, Counter
from functools import partial
import os
from multiprocessing import Manager, Pool, Value
import re
from time import time

import numpy as np
import pandas as pd
import spacy
from p_tqdm import p_uimap
from transformers import AutoTokenizer

from constants import *
from utils import *

HTML_REGEX = r'(<[a-z][^>]+>|<\/?[a-z]>)'


def sent_segment(str):
    return [x.string.strip() for x in spacy_nlp(str).sents]


def tokenize_example(text, include_bpe=False):
    split_text = re.split(HTML_REGEX, text)
    spacy_final_text = []
    sent_template = '<s> {} </s>'
    if include_bpe:
        bpe_tok_ct = 0
        bpe_final_text = []
    spacy_tok_ct = 0
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue
        if i > 0 and split_text[i - 1].strip() == '<h>':
            if include_bpe:
                bpe_final_text.append(str)
            spacy_final_text.append(str)
        elif re.search(HTML_REGEX, str) is None:
            sents = sent_segment(str) if len(str) > 50 else [str]
            for sent in sents:
                if include_bpe:
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
            if include_bpe:
                bpe_final_text.append(str)
            spacy_final_text.append(str)
    spacy_final_str = re.sub(r'\s+', ' ', ' '.join(spacy_final_text).strip())
    if include_bpe:
        bpe_final_str = re.sub(r'\s+', ' ', ' '.join(bpe_final_text).strip())
        return bpe_tok_ct, bpe_final_str, spacy_tok_ct, spacy_final_str
    return spacy_tok_ct, spacy_final_str


def tokenize_mrn(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)

    examples_df.dropna(inplace=True)
    assert len(examples_df) > 0

    spacy_source_toks, spacy_target_toks = [], []
    spacy_source_n, spacy_target_n = [], []
    for row in examples_df.to_dict('records'):
        ssn, sst = tokenize_example(row['source_str'])
        stn, stt = tokenize_example(row['target_str'])

        spacy_source_toks.append(sst)
        spacy_target_toks.append(stt)
        spacy_source_n.append(ssn)
        spacy_target_n.append(stn)

    examples_df['spacy_source_toks'] = spacy_source_toks
    examples_df['spacy_target_toks'] = spacy_target_toks
    examples_df['spacy_source_tok_ct'] = spacy_source_n
    examples_df['spacy_target_tok_ct'] = spacy_target_n
    examples_df.to_csv(examples_fn, index=False)
    return len(examples_df)


def tokenize_mrn_bpe(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)

    examples_df.dropna(inplace=True)
    assert len(examples_df) > 0

    spacy_source_toks, spacy_target_toks = [], []
    bpe_source_toks, bpe_target_toks = [], []
    spacy_source_n, spacy_target_n = [], []
    bpe_source_n, bpe_target_n = [], []
    for row in examples_df.to_dict('records'):
        bsn, bst, ssn, sst = tokenize_example(row['source_str'], include_bpe=True)
        btn, btt, stn, stt = tokenize_example(row['target_str'], include_bpe=True)

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
    return len(examples_df)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    start_time = time()
    examples = list(p_uimap(tokenize_mrn, mrns))
    print('Tokenized {} examples.'.format(sum(examples)))
    duration(start_time)
