from collections import defaultdict
import json
import math
import os
import sys

import argparse
import pandas as pd
import spacy
import scispacy
import numpy as np

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.utils import *
from preprocess.section_utils import *
from preprocess.tokenize_mrns import strip_punc


LAMBDA = -2.0


def prob(counts, globals, cui, term):
    cui_counts = counts[cui]
    denom = float(globals[cui] + len(counts[cui]))
    count = 0 if term not in cui_counts else cui_counts[term]
    numerator = count + 1
    return math.log(numerator / denom)


def tokenize(str):
    spacy_tokens = [strip_punc(x.text) for x in spacy_nlp(str)]
    return [x for x in spacy_tokens if len(x) > 0]


def get_best_prev_cost(n, c, cost_mat):
    if n == 0:
        return None, 0.0
    best_prev = None
    max_lprob = float('-inf')
    for c_prev in range(C):
        prev_lprob = 0 if n == 0 else cost_mat[c_prev, n - 1]
        if c_prev == c:
            lprob = prev_lprob
        else:
            lprob = prev_lprob + LAMBDA
        if lprob > max_lprob:
            max_lprob = lprob
            best_prev = c_prev
    return best_prev, max_lprob


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to build generate noisy semantic role labels.')
    parser.add_argument('-mini', default=False, action='store_true')

    args = parser.parse_args()

    print('Loading Spacy...')
    spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])

    mini_str = '_small' if args.mini else ''

    in_fn = os.path.join(out_dir, 'srl_packed_examples{}.csv'.format(mini_str))
    print('Reading in srl packed examples from {}'.format(in_fn))
    df = pd.read_csv(in_fn)

    out_fn = os.path.join(out_dir, 'entity', 'cui_tok_freqs.json')
    cui_tok_totals = defaultdict(int)
    with open(out_fn, 'r') as fd:
        cui_tok_freqs = json.load(fd)
    for k, v in cui_tok_freqs.items():
        cui_tok_totals[k] = sum(v.values())

    # srl_packed_target ~ generate outputs labels
    # first off, collect set of cuis and convert into labels
    examples = df.to_dict('records')

    full_labels = []
    flat_target_toks = []

    non_ent_tok_cts = 0
    ent_tok_cts = 0

    for example in tqdm(examples):
        text = example['srl_packed_target']
        mrn = example['mrn']
        account = example['account']
        cuis = set(re.findall(r'cui=(\w+) ', text))

        cui_labels = list(cuis)
        C = len(cui_labels)

        # tokenize as well as pre-label
        split_text = re.split(HTML_REGEX, example['srl_packed_target'])
        is_tag_full = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
        tokens = []
        fixed_labels = {}
        curr_state = None
        prev_state = None
        curr_cui = None
        for i, str in enumerate(split_text):
            str = str.strip()
            if len(str) == 0:
                continue

            is_tag = is_tag_full[i]
            if is_tag:
                if str[1] == '/':
                    if curr_state == 'u' or curr_state == 'f':  # these are nested
                        curr_state = prev_state
                    continue
                else:
                    prev_state = curr_state
                    curr_state = str[1]
                    assert curr_state in {'p', 'c', 'd', 'e', 'u', 'f', 'h'}
                    if curr_state == 'u':
                        curr_cui = re.findall(r'cui=(\w+) ', str)[0]
            else:
                if curr_state in {'p', 'f'}:
                    non_ent_toks = tokenize(str)
                    tokens += non_ent_toks
                    non_ent_tok_cts += len(non_ent_toks)
                elif curr_state == 'u':
                    cui_toks = tokenize(str)
                    cui_idx = cui_labels.index(curr_cui)
                    for tok_idx in range(len(tokens), len(tokens) + len(cui_toks)):
                        fixed_labels[tok_idx] = cui_idx
                    tokens += cui_toks
                    ent_tok_cts += len(cui_toks)

        N = len(tokens)
        assert N > 0
        if C == 0:
            full_labels.append(None)
            flat_target_toks.append(tokens)
            continue

        labels = []
        cost_mat = np.zeros([C, N])
        back_pointer = np.zeros([C, N], dtype=int)

        for n in range(N):
            tok = tokens[n].lower()
            if n in fixed_labels:
                fixed_label = fixed_labels[n]
                cost_mat[:, n] = float('-inf')
                best_prev, max_lprob = get_best_prev_cost(n, fixed_label, cost_mat)
                cost_mat[fixed_label, n] = max_lprob
                if best_prev is not None:
                    back_pointer[fixed_label, n] = best_prev
            else:
                for c in (range(C)):
                    emission_lprob = prob(cui_tok_freqs, cui_tok_totals, cui_labels[c], tok)
                    best_prev, max_lprob = get_best_prev_cost(n, c, cost_mat)
                    cost_mat[c, n] = max_lprob + emission_lprob
                    if best_prev is not None:
                        back_pointer[c, n] = best_prev

        # pick lowest cost
        last_idx = N - 1
        reverse_order = []
        curr_state = int(np.argmax(cost_mat[:, N - 1]))
        for n in range(N - 1, -1, -1):
            reverse_order.append(curr_state)
            curr_state = back_pointer[curr_state, n]

        order_idxs = list(reversed(reverse_order))
        labels = [cui_labels[i] for i in order_idxs]
        full_labels.append(json.dumps(labels))
        flat_target_toks.append(json.dumps(tokens))

    print('CUI token counts={}. Non-CUI token counts={}'.format(ent_tok_cts, non_ent_tok_cts))
    print('Done!  Now adding two columns and re-saving to {}'.format(in_fn))
    df['cui_labels'] = full_labels
    df['flat_spacy_target_toks'] = flat_target_toks
    df.to_csv(in_fn, index=False)
