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


def remove_empty(text_pieces, tag):
    remove_idxs = set()
    num_empty = 0
    for i, elem in enumerate(text_pieces):
        if i > 0 and elem == '</{}>'.format(tag) and text_pieces[i - 1].startswith('<{}'.format(tag)):
            remove_idxs.add(i)
            remove_idxs.add(i - 1)
            num_empty += 1
    output = []
    for i, piece in enumerate(text_pieces):
        if i in remove_idxs:
            continue
        output.append(piece)
    return output, num_empty


def decorate_w_clamp(base_fn, mrn, account, is_course=False):
    input_fn = os.path.join(out_dir, 'clamp', 'input', base_fn + '.txt')
    output_fn = os.path.join(out_dir, 'clamp', 'output', base_fn + '.txt')
    metadata_fn = os.path.join(out_dir, 'clamp', 'metadata', base_fn + '.csv')

    num_sents = 0
    num_toks = 0
    num_sections = 0

    with open(input_fn, 'r') as fd:
        text = fd.read()
    ner = []
    sent_info = []
    tok_info = []
    with open(output_fn, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            items = line.strip().split('\t')
            n = len(items)
            if items[0] == 'NamedEntity':
                row = {
                    'start': int(items[1]),
                    'end': int(items[2]),
                }
                for item in items[3:]:
                    key, value = item.split('=')
                    if key in {'semantic', 'assertion', 'cui'}:
                        row[key] = value
                    if key == 'ne':
                        row['mention'] = value
                ner.append(row)
            if items[0] == 'Sentence':
                num_sents += 1
                assert n == 4
                sent_info.append({
                    'start': int(items[1]),
                    'end': int(items[2]),
                    'section': items[3].split('=')[1]
                })
            elif items[0] == 'Token':
                assert n == 4
                tok_info.append({
                    'start': int(items[1]),
                    'end': int(items[2]),
                    'pos': items[3].split('=')[1]
                })
                num_toks += 1

    note_info = pd.read_csv(metadata_fn)
    num_notes = len(note_info)
    note_info['type'] = ['note' for _ in range(len(note_info))]
    ner = pd.DataFrame(ner)
    ner['type'] = ['entity' for _ in range(len(ner))]
    sent_info = pd.DataFrame(sent_info)
    sent_info['type'] = ['sent' for _ in range(len(sent_info))]
    tok_info = pd.DataFrame(tok_info)
    tok_info['type'] = ['token' for _ in range(len(tok_info))]

    forward_order = ['note', 'sent', 'entity', 'token']
    backward_order = list(reversed(forward_order))

    # merge all information
    positional_info = defaultdict(lambda: defaultdict(dict))

    all_sections = []

    all_records = (ner.to_dict('records') + sent_info.to_dict('records') + tok_info.to_dict('records') +
                   note_info.to_dict('records'))
    for record in all_records:
        positional_info[record['start']]['start'][record['type']] = record
        positional_info[record['end']]['end'][record['type']] = record

    text_pieces = []
    entity_metadata = []
    prev_start = 0
    curr_section = None
    all_positions = np.sort(list(positional_info.keys()))
    should_break = False
    for pos_idx in all_positions:
        text_pieces.append(text[prev_start:pos_idx].strip())
        pos_obj = positional_info[pos_idx]
        if 'end' in pos_obj:
            for annotation_type in backward_order:
                if annotation_type in pos_obj['end']:
                    end_annotate = pos_obj['end'][annotation_type]
                    if annotation_type == 'token':
                        # text_pieces.append('</t>')
                        pass
                    elif annotation_type == 'sent':
                        text_pieces.append('</s>')
                    elif annotation_type == 'entity':
                        text_pieces.append('</e>')
                    elif annotation_type == 'note':
                        text_pieces.append('</d>')
                    else:
                        raise Exception('Unknown entity type={}'.format(annotation_type))
        if 'start' in pos_obj:
            for annotation_type in forward_order:
                if annotation_type in pos_obj['start']:
                    start_annotate = pos_obj['start'][annotation_type]
                    if annotation_type == 'token':
                        # text_pieces.append('<t>')
                        pass
                    elif annotation_type == 'sent':
                        section = start_annotate['section']
                        if not section == curr_section:
                            if is_course and curr_section == 'hospital_course' and not section == 'hospital_course':
                                should_break = True
                                break
                            if curr_section is not None:
                                text_pieces.append('</h>')
                            text_pieces.append('<h section={}>'.format(section))
                            curr_section = section
                            num_sections += 1
                            all_sections.append(section.strip())
                        text_pieces.append('<s>')
                    elif annotation_type == 'entity':
                        mid = len(entity_metadata)
                        text_pieces.append('<e id={}>'.format(mid))
                        start_annotate['id'] = mid
                        entity_metadata.append(start_annotate)
                    elif annotation_type == 'note':
                        note_id = start_annotate['note_id']
                        text_pieces.append('<d note_id={}>'.format(note_id.strip()))
                    else:
                        raise Exception('Unknown entity type={}'.format(annotation_type))

        prev_start = pos_idx
        if should_break:
            break

    text_pieces.append('</h>')
    text_pieces.append('</d>')
    entity_metadata = pd.DataFrame(entity_metadata)

    # Remove duplicate sentences
    sent_inventory = set()

    # Remove sections which are empty after removing duplicate sentences
    remove_idxs = []
    num_dup_sents = 0
    num_dup_toks = 0
    sent_start_idx = None
    for i, elem in enumerate(text_pieces):
        if elem == '<s>':
            sent_start_idx = i
        elif elem == '</s>':
            sent_end_idx = i
            extracted_sent = re.sub(r'\W', '', ''.join(final_text[sent_start_idx + 1:sent_end_idx]).lower())
            if extracted_sent in sent_inventory:
                remove_idxs += list(range(sent_start_idx, sent_end_idx + 1))
                num_dup_sents += 1
                num_dup_toks += sent_end_idx - sent_start_idx - 1
            else:
                sent_inventory.add(extracted_sent)

    print('{} out of {} were duplicated'.format(num_dup_sents, num_sents))
    num_sents -= num_dup_sents
    num_toks -= num_dup_toks
    text_pieces_no_dup_sents = []
    remove_idxs = set(remove_idxs)
    for idx, st in enumerate(text_pieces):
        if idx in remove_idxs:
            continue
        text_pieces_no_dup_sents.append(st)

    # Remove empty sections and empty notes (after removing duplicate sections)
    text_pieces_no_dup_sections, num_empty_sections = remove_empty(text_pieces_no_dup_sents, 'h')
    num_sections -= num_empty_sections

    final_text_pieces, num_empty_notes = remove_empty(text_pieces_no_dup_sections,'d')
    num_notes -= num_empty_notes

    final_text_str = ' '.join(final_text_pieces)
    final_text_str = re.sub(r'\s+', ' ', final_text_str)

    counts = {
        'mrn': str(mrn),
        'account': str(account),
        'is_target': is_course,
        'num_notes': num_notes,
        'num_sections': num_sections,
        'num_sents': num_sents,
        'num_toks': num_toks,
    }

    return final_text_str, entity_metadata, counts, Counter(all_sections)


# def sent_segment(str, sentencizer=None):
#     sent_lists = [x.strip() for x in re.split(LIST_REGEX, str) if len(x.strip()) > 0]
#
#     spacy_sents = []
#     for sent in sent_lists:
#         for newline_sent in re.split(NEWLINE_REGEX, sent):
#             if len(newline_sent.strip()) > 0:
#                 if sentencizer is None:
#                     spacy_sents += [x.string.strip() for x in spacy_nlp(newline_sent).sents]
#                 else:
#                     spacy_sents += [x.string.strip() for x in sentencizer(newline_sent).sents]
#
#     sub_sents = []
#     for sent in spacy_sents:
#         sent_len = len(sent.split(' '))
#         if sent_len > 20:
#             sub_sents += [x.strip() for x in re.split(SUB_HEADERS, sent) if len(x.strip()) > 0]
#         else:
#             sub_sents.append(sent)
#
#     shorter_sents = []
#     for sent in sub_sents:
#         sent_len = len(sent.split(' '))
#         if sent_len > 30:  # look for other delimiters
#             shorter_sents += [x.strip() for x in re.split(LONG_DELIMS, sent) if len(x.strip()) > 0]
#         else:
#             shorter_sents.append(sent)
#
#     return shorter_sents


# def strip_punc(str):
#     if len(str) == 1:
#         return str
#     return str.strip(string.punctuation)


# def break_up_big_sents(str):
#     return sent_segment(str) if len(str) > 50 else [str]


def process_clamp(mrn):
    mrn_dir = os.path.join(out_dir, 'mrn', mrn)
    examples_fn = os.path.join(mrn_dir, 'examples.csv')

    entity_fn = os.path.join(mrn_dir, 'entity.csv')
    examples_df = pd.read_csv(examples_fn)
    assert len(examples_df.dropna()) == len(examples_df) > 0

    entity_df, all_source_counts, all_target_counts, all_section_counts = [], [], [], Counter()

    output = defaultdict(list)
    for row in examples_df.to_dict('records'):
        account = row['account']
        source_clamp_fn = '{}_{}_{}'.format(str(mrn), str(account), 'source')
        decorated_source, source_metadata, source_counts, section_counts = decorate_w_clamp(
            source_clamp_fn, mrn, account, is_course=False)
        target_clamp_fn = '{}_{}_{}'.format(str(mrn), str(account), 'target')
        decorated_target, target_metadata, target_counts, _ = decorate_w_clamp(
            target_clamp_fn, mrn, account, is_course=True)

        entity_df += source_metadata
        entity_df += target_metadata
        all_section_counts += section_counts

        output['decorated_source'] = decorated_source
        output['decorated_target'] = decorated_target

    entity_df = pd.DataFrame(entity_df)
    entity_df.to_csv(entity_fn, index=False)
    for k, v in output.items():
        examples_df[k] = v

    # Generally, this means we have a repeated hospital course for some reason
    examples_df.drop_duplicates(subset=['target_tok_ct'], inplace=True)
    examples_df.to_csv(examples_fn, index=False)
    return len(examples_df), all_source_counts, all_target_counts, all_section_counts


if __name__ == '__main__':
    # print('Loading scispacy')
    # spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    # spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))
    # print('Ready to tokenize!')

    _, _, mrns = get_mrn_status_df('valid_example')
    n = len(mrns)
    print('Processing {} mrns'.format(n))
    outputs = list(p_uimap(process_clamp, mrns, num_cpus=0.8))
    examples = [x[0] for x in outputs]
    source_sent_lens = np.array(list(itertools.chain(*[x[1] for x in outputs])))
    target_sent_lens = np.array(list(itertools.chain(*[x[2] for x in outputs])))
    num_examples = sum(examples)
    too_big_ct = sum([x[3] for x in outputs])
    print('Tokenized {} examples. {} were flagged as too big'.format(num_examples, too_big_ct))

    print('Average source sentence length: {}'.format(source_sent_lens.mean()))
    print('Average target sentence length: {}'.format(target_sent_lens.mean()))
