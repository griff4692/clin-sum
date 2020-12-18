from collections import defaultdict
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

sys.path.insert(0, os.path.expanduser('~/clin-sum'))
from preprocess.constants import *
from preprocess.section_utils import HTML_REGEX
from preprocess.tokenize_mrns import sent_segment
from preprocess.utils import *


def decile(x):
    for start_idx in np.arange(0.0, 1.0, 0.1):
        end_idx = start_idx + 0.1
        if start_idx <= x < end_idx:
            return '{}-{}'.format(str(round(start_idx, 1)), str(round(end_idx, 1)))
    return '0.9-1.0'


def coverage(doc_order, coverages, doc_to_ent, target_n):
    covered = set()
    docs_used = len(doc_order)
    for doc_num, doc_idx in enumerate(doc_order):
        ents = doc_to_ent[doc_idx]
        prev_size = len(covered)
        covered = covered.union(ents)
        now_size = len(covered)
        ratio = (now_size - prev_size) / float(target_n)
        doc_position = decile(doc_num / float(num_docs - 1))
        coverages[doc_position].append(ratio)
        if len(covered) == target_n:
            docs_used = min(docs_used, doc_num + 1)
    return docs_used


def aggregate(coverages):
    coverages_agg = []
    for decile, ratios in coverages.items():
        avg = np.array(ratios).mean()
        coverages_agg.append({
            'decile': decile,
            'coverage': avg,
            'examples': len(ratios),
        })

    return coverages_agg


if __name__ == '__main__':
    print('Loading entity dataframe...')
    examples = pd.read_csv(os.path.join(out_dir, 'full_examples.csv'))
    records = examples.to_dict('records')

    print('Loading Spacy...')
    sentencizer = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))
    num_records = len(records)

    docs_used = []
    positions = []
    skipped = 0
    forward_coverages = defaultdict(list)
    oracle_coverages = defaultdict(list)
    backward_coverages = defaultdict(list)

    for record_idx in tqdm(range(num_records)):
        record = records[record_idx]
        mrn = record['mrn']
        account = record['account']
        ents_fn = os.path.join(out_dir, 'entity', 'entities', '{}_{}.csv'.format(str(mrn), str(account)))

        if not os.path.exists(ents_fn):
            # print('No entities for mrn={}, account={}.'.format(mrn, account))
            continue

        ents_df = pd.read_csv(ents_fn)
        source_ents = ents_df[ents_df['is_source']]
        target_ents = ents_df[ents_df['is_target']]

        overlapping_cuis = set(source_ents['cui']).intersection(set(target_ents['cui']))
        source_ents_rel = source_ents[source_ents['cui'].isin(overlapping_cuis)]
        target_n = len(overlapping_cuis)

        if target_n == 0:
            skipped += 1
            continue

        max_sent_idx = source_ents_rel['sent_idx'].max()

        actual_sent_idxs = []
        doc_ids = []
        section_ids = []
        text = record['source_str']

        split_text = re.split(HTML_REGEX, text)
        is_tag = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
        curr_doc_id = -1
        curr_section_id = -1
        sent_start_idx = 0
        num_docs = 0
        for i, text_str in enumerate(split_text):
            text_str = text_str.strip()
            if len(text_str) == 0:
                continue
            if is_tag[i]:
                if text_str.startswith('<p'):
                    num_sents = len(sent_segment(split_text[i + 1].strip(), sentencizer=sentencizer))
                    doc_ids += [curr_doc_id] * num_sents
                    section_ids += [curr_section_id] * num_sents
                    actual_sent_idxs += list(range(sent_start_idx, sent_start_idx + num_sents))
                    sent_start_idx += num_sents
                elif text_str.startswith('<d'):
                    num_docs += 1
                    curr_doc_id += 1
                    curr_section_id = -1
                    sent_start_idx = 0
                elif text_str.startswith('<h'):
                    curr_section_id += 1
                else:
                    assert text_str.startswith('<e') or text_str[1] == '/'

        doc_lens = [0 for _ in range(len(doc_ids))]
        for doc_id in doc_ids:
            doc_lens[doc_id] += 1

        # assert len(doc_ids) > max_sent_idx
        if len(doc_ids) <= max_sent_idx:
            print('Sentence alignment error.  Skipping for now but come back to this, please!')
            print(len(doc_ids), max_sent_idx)
            skipped += 1
            continue

        doc_to_ent = [set() for _ in range(num_docs)]
        docs_by_num_ents = [0 for _ in range(num_docs)]
        for entity in source_ents_rel.to_dict('records'):
            sent_idx = entity['sent_idx']
            actual_sent_idx = actual_sent_idxs[sent_idx]
            cui = entity['cui']
            doc_idx = doc_ids[sent_idx]
            doc_len = doc_lens[doc_idx]
            doc_to_ent[doc_idx].add(cui)
            positions.append(actual_sent_idx / max(1.0, float(doc_len - 1)))
        for doc_idx in range(num_docs):
            docs_by_num_ents[doc_idx] = len(doc_to_ent[doc_idx])
        doc_order = np.argsort(docs_by_num_ents)[::-1]

        if num_docs == 1:
            continue

        forward_docs_used = coverage(list(range(num_docs)), forward_coverages, doc_to_ent, target_n)
        backward_docs_used = coverage(list(reversed(range(num_docs))), backward_coverages, doc_to_ent, target_n)
        oracle_docs_used = coverage(doc_order, oracle_coverages, doc_to_ent, target_n)

        docs_used.append({
            'forward_docs_used': forward_docs_used,
            'forward_docs_used_perc': forward_docs_used / float(num_docs),
            'backward_docs_used': backward_docs_used,
            'backward_docs_used_perc': backward_docs_used / float(num_docs),
            'oracle_docs_used': oracle_docs_used,
            'oracle_docs_used_perc': oracle_docs_used / float(num_docs),
        })

    docs_used = pd.DataFrame(docs_used)
    agg_docs_used = []
    for col in list(docs_used.columns):
        col_mean = docs_used[col].mean()
        col_std = docs_used[col].std()
        col_median = np.median(docs_used[col])

        agg_docs_used.append({'stat': col + '_mean', 'value': col_mean})
        agg_docs_used.append({'stat': col + '_std', 'value': col_std})
        agg_docs_used.append({'stat': col + '_median', 'value': col_median})

    agg_docs_used = pd.DataFrame(agg_docs_used)
    print('Caveat! Skipped {} out of {} examples.'.format(skipped, len(records)))
    agg_docs_used.to_csv('data/docs_used.csv', index=False)

    forward_chronology = pd.DataFrame(aggregate(forward_coverages))
    forward_chronology.to_csv('data/forward_chronology_coverage.csv', index=False)

    backward_chronology = pd.DataFrame(aggregate(backward_coverages))
    backward_chronology.to_csv('data/backward_chronology_coverage.csv', index=False)

    oracle_chronology = pd.DataFrame(aggregate(oracle_coverages))
    oracle_chronology.to_csv('data/oracle_chronology_coverage.csv', index=False)

    df = pd.DataFrame({'positions': positions})
    df.to_csv('data/source_relative_positions.csv', index=False)
