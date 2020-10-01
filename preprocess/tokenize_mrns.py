from collections import defaultdict, Counter
import itertools
from functools import partial
import os
import re
import string
import sys
from time import time

import numpy as np
import pandas as pd
from p_tqdm import p_uimap
import spacy

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.section_utils import resolve_course
from preprocess.utils import *

HTML_REGEX = r'(<[a-z][^>]+>|<\/?[a-z]>)'
NEWLINE_REGEX = r'\n+[-.#]+'
LIST_REGEX = r'\s+\d\)|\d\)\s+|\s+\d\.\s+'
LONG_DELIMS = r'\-+ | \-+'
SUB_HEADERS = r' (?=[A-z]+:)'


def sent_segment(str, sentencizer=None):
    sent_lists = [x.strip() for x in re.split(LIST_REGEX, str) if len(x.strip()) > 0]

    spacy_sents = []
    for sent in sent_lists:
        for newline_sent in re.split(NEWLINE_REGEX, sent):
            if len(newline_sent.strip()) > 0:
                if sentencizer is None:
                    spacy_sents += [x.string.strip() for x in spacy_nlp(newline_sent).sents]
                else:
                    spacy_sents += [x.string.strip() for x in sentencizer(newline_sent).sents]

    sub_sents = []
    for sent in spacy_sents:
        sent_len = len(sent.split(' '))
        if sent_len > 20:
            sub_sents += [x.strip() for x in re.split(SUB_HEADERS, sent) if len(x.strip()) > 0]
        else:
            sub_sents.append(sent)

    shorter_sents = []
    for sent in sub_sents:
        sent_len = len(sent.split(' '))
        if sent_len > 30:  # look for other delimiters
            shorter_sents += [x.strip() for x in re.split(LONG_DELIMS, sent) if len(x.strip()) > 0]
        else:
            shorter_sents.append(sent)

    return shorter_sents


def strip_punc(str):
    if len(str) == 1:
        return str
    return str.strip(string.punctuation)


def tokenize_example(text):
    split_text = re.split(HTML_REGEX, text)
    spacy_final_text = []
    sent_template = '<s> {} </s>'
    spacy_tok_ct = 0
    sent_lens = []
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue
        if i > 0 and split_text[i - 1].strip() == '<h>':
            spacy_final_text.append(str)
        elif re.search(HTML_REGEX, str) is None:
            sents = sent_segment(str) if len(str) > 50 else [str]
            for sent in sents:
                # if 'authored' in sent.lower() or 'signed' in sent.lower() or 'last updated' in sent.lower():
                #     continue
                spacy_tokens = [strip_punc(x.text) for x in spacy_nlp(sent)]
                spacy_tokens = [x for x in spacy_tokens if len(x) > 0]
                sent_len = len(spacy_tokens)
                spacy_tok_ct += sent_len
                spacy_token_str = ' '.join(spacy_tokens)
                spacy_sent_str = sent_template.format(spacy_token_str)
                spacy_final_text.append(spacy_sent_str)
                sent_lens.append(sent_len)
        else:
            spacy_final_text.append(str)
    spacy_final_str = re.sub(r'\s+', ' ', ' '.join(spacy_final_text).strip())
    return spacy_tok_ct, sent_lens, spacy_final_str


def tokenize_mrn(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')
    examples_df = pd.read_csv(examples_fn)
    assert len(examples_df.dropna()) == len(examples_df) > 0
    source_sent_lens, target_sent_lens = [], []

    spacy_source_toks, spacy_target_toks = [], []
    spacy_source_n, spacy_target_n = [], []
    is_too_big = []
    too_big_ct = 0
    for row in examples_df.to_dict('records'):
        ssn, ssl, sst = tokenize_example(row['source_str'])
        stn, tsl, stt = tokenize_example(resolve_course(row['target_str']))
        source_sent_lens += ssl
        target_sent_lens += tsl

        too_big = ssn >= MAX_SOURCE_TOK_CT or stn >= MAX_TARGET_TOK_CT
        is_too_big.append(too_big)
        if too_big:
            too_big_ct += 1
        spacy_source_toks.append(sst)
        spacy_target_toks.append(stt)
        spacy_source_n.append(ssn)
        spacy_target_n.append(stn)

    examples_df['spacy_source_toks'] = spacy_source_toks
    examples_df['spacy_target_toks'] = spacy_target_toks
    examples_df['spacy_source_tok_ct'] = spacy_source_n
    examples_df['spacy_target_tok_ct'] = spacy_target_n
    examples_df['is_too_big'] = is_too_big
    # Generally, this means we have a repeated hospital course for some reason
    examples_df.drop_duplicates(subset=['spacy_target_tok_ct'], inplace=True)
    examples_df.to_csv(examples_fn, index=False)
    return len(examples_df), source_sent_lens, target_sent_lens, too_big_ct


if __name__ == '__main__':
    print('Loading scispacy')
    spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))
    print('Ready to tokenize!')

    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    outputs = list(p_uimap(tokenize_mrn, mrns, num_cpus=0.8))
    examples = [x[0] for x in outputs]
    source_sent_lens = np.array(list(itertools.chain(*[x[1] for x in outputs])))
    target_sent_lens = np.array(list(itertools.chain(*[x[2] for x in outputs])))
    num_examples = sum(examples)
    too_big_ct = sum([x[3] for x in outputs])
    print('Tokenized {} examples. {} were flagged as too big'.format(num_examples, too_big_ct))

    print('Average source sentence length: {}'.format(source_sent_lens.mean()))
    print('Average target sentence length: {}'.format(target_sent_lens.mean()))
