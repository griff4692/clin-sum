from collections import Counter, defaultdict
import json
import itertools
import json
import math
import os
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import argparse
from lexrank import STOPWORDS
from lexrank.utils.text import tokenize
import numpy as np
import pandas as pd
from p_tqdm import p_uimap
from scipy.stats import pearsonr

from evaluations.rouge import compute, prepare_str_for_rouge
from extractive.lexrank.lexrank_model import LexRank
from preprocess.constants import out_dir
from preprocess.section_utils import sents_from_html
from preprocess.utils import get_records


TARGET_TOK_CT = 195


def stringify_list(a):
    return ','.join([str(x) for x in a])


def aggregate(source_str):
    return ' '.join(sents_from_html(source_str, convert_lower=True))


def build(sent_order, sents, record, target_tok_ct=TARGET_TOK_CT):
    summary_sents = []
    sum_len = 0
    sent_lens = []

    assert len(sents) > 0
    # build summaries by total length
    for sent_idx in sent_order:
        sent = sents[sent_idx]
        if sent in summary_sents:
            continue
        this_len = len(sent.split(' '))
        if sum_len + this_len > target_tok_ct and not len(summary_sents) == 0:
            break
        sent_lens.append(this_len)
        summary_sents.append(sent)
        sum_len += this_len
    prediction = ' <s> '.join(summary_sents).strip()

    sent_order_used = sent_order[:len(summary_sents)]
    target_sents = sents_from_html(record['spacy_target_toks'], convert_lower=True)
    n = len(target_sents)
    reference = ' <s> '.join(target_sents).strip()
    ref_len = len(reference.split(' ')) - n  # subtract pseudo sentence tokens

    return {
        'mrn': record['mrn'],
        'account': record['account'],
        'reference': reference,
        'prediction': prediction,
        'ref_len': ref_len,
        'sum_len': sum_len,
        'sent_order': stringify_list(sent_order_used)
    }, sent_lens


def describe(arr):
    print('Mean={}, Median={}. Std={}.'.format(arr.mean(), np.median(arr), arr.std()))


def compute_lr_stats(record):
    sents = list(set(sents_from_html(record['spacy_source_toks'], convert_lower=True)))
    reference = prepare_str_for_rouge(record['spacy_target_toks'].lower())
    lr_scores = np.array(list(lxr.rank_sentences(
        sents,
        threshold=0.1,
        fast_power_method=True,
    )))

    n = len(sents)
    predictions = [prepare_str_for_rouge(s) for s in sents]
    references = [reference for _ in range(n)]
    rouge_types = ['rouge1', 'rouge2']
    outputs = compute(predictions, references, rouge_types=rouge_types, use_aggregator=False)
    r_scores = np.array(
        [sum([outputs[t][i].fmeasure for t in rouge_types]) / float(len(rouge_types)) for i in range(n)])
    return list(zip(lr_scores, r_scores))


def compute_lr(record):
    sents = sents_from_html(record['spacy_source_toks'], convert_lower=True)
    unique_sents = list(set(sents))
    sent_scores = np.array(list(lxr.rank_sentences(
        sents,
        threshold=0.1,
        fast_power_method=True,
    )))

    frac_uniq = len(unique_sents) / float(len(sents))

    sent_scores_deduped = np.array(list(lxr.rank_sentences(
        unique_sents,
        threshold=0.1,
        fast_power_method=True,
    )))

    sent_order = np.argsort(-sent_scores)
    sent_order_deduped = np.argsort(-sent_scores_deduped)

    o1, sent_lens = build(sent_order, sents, record)
    o2, _ = build(sent_order_deduped, unique_sents, record)
    return o1, o2, np.array(sent_lens).mean(), frac_uniq


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to run LexRank')
    parser.add_argument('-compute_stats', default=False, action='store_true')

    args = parser.parse_args()

    idf_fn = os.path.join('data/idf.json')

    stopwords = STOPWORDS['en']
    stopwords = stopwords.union(set([x for x in punctuation]))

    if os.path.exists(idf_fn):
        print('Loading pre-computed IDF...')
        with open(idf_fn, 'r') as fd:
            idf_obj = json.load(fd)

        idf_score_dd = defaultdict(lambda: idf_obj['default'])
        print('Default IDF={}'.format(idf_obj['default']))
        for k, v in idf_obj['idf_score'].items():
            idf_score_dd[k] = v
        lxr = LexRank(idf_score=idf_score_dd, stopwords=stopwords, default=idf_obj['default'])
    else:
        print('Computing IDF...')
        train_df = get_records(split='train', mini=False)
        print('Building LexRank...')

        train_docs = train_df['spacy_target_toks'].apply(aggregate)
        lxr = LexRank(train_docs, stopwords=stopwords)

        idf_obj = {
            'idf_score': dict(lxr.idf_score),
            'default': lxr.default
        }

        print('Saving IDF so we can don\'t need to recompute it...')
        with open(idf_fn, 'w') as fd:
            json.dump(idf_obj, fd)

    validation_records = get_records(split='validation', mini=False).to_dict('records')

    if args.compute_stats:
        outputs = list(itertools.chain(*list(p_uimap(compute_lr_stats, validation_records, num_cpus=0.8))))
        lr_scores = [o[0] for o in outputs]
        r_scores = [o[1] for o in outputs]
        corel = pearsonr(lr_scores, r_scores)
        print('Pearson correlation of LR and R12={}. p-value={}'.format(corel[0], corel[1]))
    else:
        outputs = list(p_uimap(compute_lr, validation_records, num_cpus=0.8))

        exp_str = 'lr'
        out_fn = os.path.join(out_dir, 'predictions', '{}_validation.csv'.format(exp_str))

        output = [o[0] for o in outputs]
        output_df = pd.DataFrame(output)
        n = len(output_df)
        print('Saving {} predictions to {}'.format(n, out_fn))
        output_df.to_csv(out_fn, index=False)
        print('To evaluate, run: cd ../../evaluations && python rouge.py --experiment {}'.format(exp_str))

        output_deduped = [o[1] for o in outputs]
        output_deduped_df = pd.DataFrame(output_deduped)
        exp_deduped_str = 'lr_deduped'
        out_deduped_fn = os.path.join(out_dir, 'predictions', '{}_validation.csv'.format(exp_deduped_str))
        print('Saving {} predictions to {}'.format(n, out_deduped_fn))
        output_deduped_df.to_csv(out_deduped_fn, index=False)
        print('To evaluate, run: cd ../../evaluations && python rouge.py --experiment {}'.format(exp_deduped_str))

        sent_lens = np.array([o[2] for o in outputs])
        uniq_fraqs = np.array([o[3] for o in outputs])

        print('Sentence Lengths...')
        describe(sent_lens)

        print('Unique Sentence Proportion')
        describe(uniq_fraqs)
