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
    coverages = defaultdict(list)
    backward_coverages = defaultdict(list)

    for record_idx in tqdm(range(num_records)):
        record = records[record_idx]
        mrn = record['mrn']
        account = record['account']
        ents_fn = os.path.join(out_dir, 'entity', 'entities', '{}_{}.csv'.format(str(mrn), str(account)))

        if not os.path.exists(ents_fn):
            print('No entities for mrn={}, account={}.'.format(mrn, account))
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
            positions.append(actual_sent_idx / float(doc_len - 1))
        for doc_idx in range(num_docs):
            docs_by_num_ents[doc_idx] = len(doc_to_ent[doc_idx])
        doc_order = np.argsort(docs_by_num_ents)[::-1]

        need = []
        covered = set()
        num_docs_used = 0
        for doc_idx in doc_order:
            covered = covered.union(doc_to_ent[doc_idx])
            num_docs_used += 1
            if len(covered) == target_n:
                break

        docs_used.append({
            'num_docs_used': num_docs_used,
            'num_docs': target_n,
            'percent_used': num_docs_used / float(target_n)
        })

        covered = set()
        for doc_idx in range(num_docs):
            ents = doc_to_ent[doc_idx]
            prev_size = len(covered)
            covered = covered.union(doc_to_ent[doc_idx])
            now_size = len(covered)
            ratio = (now_size - prev_size) / float(target_n)
            coverages[doc_idx].append(ratio)
            if len(covered) == target_n:
                break

        covered = set()
        for doc_idx in range(num_docs - 1, -1, -1):
            ents = doc_to_ent[doc_idx]
            prev_size = len(covered)
            covered = covered.union(doc_to_ent[doc_idx])
            now_size = len(covered)
            ratio = (now_size - prev_size) / float(target_n)
            backward_coverages[doc_idx].append(ratio)
            if len(covered) == target_n:
                break

    docs_used = pd.DataFrame(docs_used)
    print(docs_used['num_docs_used'].mean())
    print(docs_used['num_docs_used'].std())
    print(docs_used['percent_used'].mean())
    print(docs_used['percent_used'].std())
    print('Caveat! Skipped {} out of {} examples.'.format(skipped, len(records)))
    docs_used.to_csv('data/docs_used.csv', index=False)

    chronology = []
    for doc_idx, ratios in coverages.items():
        avg = np.array(ratios).mean()
        chronology.append({
            'doc_idx': doc_idx,
            'coverage': avg,
            'examples': len(ratios),
        })

    chronology = pd.DataFrame(chronology)
    chronology['doc_num'] = chronology['doc_idx'].apply(lambda x: x + 1)
    chronology.sort_values(by='doc_num', inplace=True)
    chronology[['doc_num', 'coverage', 'examples']].to_csv('data/chronology_coverage.csv', index=False)

    backwards = []
    for doc_idx, ratios in backward_coverages.items():
        avg = np.array(ratios).mean()
        backwards.append({
            'doc_idx': doc_idx,
            'coverage': avg,
            'examples': len(ratios),
        })

    backwards = pd.DataFrame(backwards)
    backwards['doc_num'] = backwards['doc_idx'].apply(lambda x: x + 1)
    backwards.sort_values(by='doc_num', ascending=False, inplace=True)
    backwards[['doc_num', 'coverage', 'examples']].to_csv('data/backward_chronology_coverage.csv', index=False)

    df = pd.DataFrame({'positions': positions})
    df.to_csv('data/source_relative_positions.csv', index=False)
