import itertools
import os
import re
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess.constants import *
from preprocess.utils import *

# Building Blocks
BEGINNING = r'(?:^|\s{4,}|\t|\n)'
ALL_CAPS = r'[A-Z][A-Z0-9/() ]{0,' + str(MAX_HEADER_LEN - 2) + '}[A-Z)]'
MIXED_CAPS = r'[A-Z][A-z0-9/() ]{0,' + str(MAX_HEADER_LEN - 2) + '}[A-z)]'
NEWLINE = r'\s*\n'
COLON = r':+'

# 3 Different header styles
MIXED_CAPS_COURSE = r'(?:Hospital|ER|CLINIC|OR|ED|ICU|MICU|SICU|Post[- ]?[Oo]p)? ?Course(?!,)'
LOWER_CASE_COURSE = r'(?:hospital|er|clinic|or|ed|icu|micu|sicu|post[- ]?op)? ?course:+'
MIXED_CAPS_COLON = r'{}{}'.format(MIXED_CAPS, COLON)
BEGIN_MIXED_CAPS_COLON = r'{}{}{}'.format(BEGINNING, MIXED_CAPS, COLON)
BEGIN_SHORT_MIXED_CAPS_NEWLINE = r'{}{}{}'.format(BEGINNING, r'[A-Z][A-z]+\s?[A-z]+', NEWLINE)
ALL_CAPS_NEWLINE = r'{}{}'.format(ALL_CAPS, NEWLINE)
BEGIN_ALL_CAPS_NEWLINE = r'{}{}{}'.format(BEGINNING, ALL_CAPS, NEWLINE)
ALL_CAPS_COLON = r'{}{}'.format(ALL_CAPS, COLON)

HTML_REGEX = r'(<[a-z][^>]+>|<\/?[a-z]>)'
HEADER_REGEX = r'({}|{}|{}|{}|{})'.format(
    BEGIN_MIXED_CAPS_COLON, BEGIN_ALL_CAPS_NEWLINE, ALL_CAPS_COLON, MIXED_CAPS_COURSE, LOWER_CASE_COURSE)
ONLY_INLINE_HEADER_REGEX = r'({}|{}|{}|{})'.format(
    MIXED_CAPS_COLON, ALL_CAPS_NEWLINE, ALL_CAPS_COLON, BEGIN_SHORT_MIXED_CAPS_NEWLINE)
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'
systems = ['resp', 'fen', 'gi', 'cv', 'heme', 'id', 'neuro', 'social', 'endo', 'other', 'renal', 'psych']
irrelevant_sections = [
    r'labs?', r'pmh', r'past medical (history|hx)', r'history of (the )?present illness', r'hpi', r'chief complaint',
    r'exam', r'medication', r'smh', r'social (history|hx)', r'\bpsh\b', r'family', r'complications?', r'plans?',
    r'follow[- ]?up', r'contact information', r'instructions', r'date', r'surgical history', r'impression', r'doctor',
    r'patient', r'\bmh\b', r'\bsh\b', r'\bfh\b', r'assessment', r'handout', r'\bdispo', r'\bpe\b', r'\bimag',
    r'maintenance', r'measure', r'\bnote\b', r'\bvital', r'(on|at)? ?discharge', r'path(ology)?', r'authored',
    r'assistant', r'allerg', r'physician', r'surgeon', r'appointment', r'schedul'
]


def same_text(a, b):
    return re.sub(r'\W+', '', a.lower()) == re.sub(r'\W+', '', b.lower())


def clean(text):
    cleaned = []
    for line in text.split('\n'):
        line_strip = ''.join([x if ord(x) < 128 else '' for x in line]).strip()
        line_strip = re.sub(r'[-_.<\s]{10,}', ' ', line_strip)
        line_strip = re.sub(r'\s+\[retrieved for.*\]', '', line_strip)
        line_strip = re.sub(r'[<>]+', ' ', line_strip)  # this is a special character for us (don't confuse)
        line_strip = re.sub(r'\s+', ' ', line_strip)
        if len(line_strip) > 0:
            cleaned.append(line_strip)
    return '\n'.join(cleaned)


def _is_resolved_relevant(text, is_header, is_relevant_section):
    lower_str = text.lower()
    trunc_idx = len(lower_str)

    unacceptable_subsection_regex = r'{}'.format(
        BEGINNING + '[A-z/ ]{0,' + str(MAX_HEADER_LEN // 2) + '}(' +
        '|'.join(irrelevant_sections) + ')[A-z/ ]{0,' + str(MAX_HEADER_LEN // 2) + '}')
    is_relevant_subsection = re.match(unacceptable_subsection_regex, lower_str[:min(trunc_idx, MAX_HEADER_LEN)]) is None

    # To add: either it does not start with a section header or it is an allowable section header
    to_add = (not is_header) or is_relevant_section or is_relevant_subsection

    # do we keep going? to_add and there are no black listed subsections hidden here
    inline_header = re.findall(ONLY_INLINE_HEADER_REGEX, text, flags=re.M)
    has_irrelevant_subs = False
    for header in inline_header:
        if 'course' in header.lower():
            continue
        if re.search('|'.join(irrelevant_sections), header.lower()) is not None:
            trunc_idx = text.index(header)
            has_irrelevant_subs = True
            break
    to_continue = to_add and not has_irrelevant_subs

    return to_add, to_continue, trunc_idx


def section_split(text):
    sectioned_text = re.split(HEADER_REGEX, text, flags=re.M)
    sectioned_text = [x.lstrip() for x in sectioned_text if len(x.lstrip()) > 0]
    is_header_arr = list(map(lambda x: re.match(HEADER_REGEX, x, re.M) is not None, sectioned_text))
    return sectioned_text, is_header_arr


def sectionize(text):
    sectioned_text, is_header_arr = section_split(text)
    output_str = ''
    n = len(sectioned_text)
    for i, (is_header, t) in enumerate(zip(is_header_arr, sectioned_text)):
        template = '<h> {} </h>' if is_header and len(t) < MAX_HEADER_LEN else '<p> {} </p>'
        is_next_header = i < n - 1 and is_header_arr[i + 1]
        if is_header and is_next_header:
            continue
        output_str += template.format(t) + ' '
    return output_str.strip()


def extract_hospital_course(text):
    sectioned_text, is_header_arr = section_split(text)
    is_relevant_section = list(map(
        lambda x: x[0] and x[1] is not None and 'course' in x[1].lower() and len(x[1]) < MAX_HEADER_LEN,
        zip(is_header_arr, sectioned_text)
    ))

    for i, rel in enumerate(is_relevant_section):
        is_dup = (rel and i < len(is_relevant_section) - 1 and is_relevant_section[i + 1]
                  and same_text(sectioned_text[i], sectioned_text[i + 1]))
        if is_dup:
            is_relevant_section[i] = False

    course_idxs = np.where(is_relevant_section)[0]
    strs = []
    covered = set()
    for relevant_start_idx in course_idxs:
        if relevant_start_idx in covered:
            continue
        course_str = ''
        body_len = 0
        for i in range(relevant_start_idx, len(sectioned_text)):
            covered.add(i)
            t = sectioned_text[i].strip()
            is_header = is_header_arr[i]
            template = '<h> {} </h>' if is_header and len(t) < MAX_HEADER_LEN else '<p> {} </p>'

            is_relevant = is_relevant_section[i]
            to_add, to_continue, trunc_idx = _is_resolved_relevant(t, is_header, is_relevant)
            if to_add:
                truncated = t[:trunc_idx].strip()
                if len(truncated) > 0:
                    course_str += template.format(truncated) + ' '
                    if not is_header:
                        body_len += len(truncated)
            if not to_continue:
                break
        course_str = course_str.strip()
        if body_len >= MIN_TARGET_LEN:
            strs.append(course_str)
    strs = list(set(strs))
    return ' '.join(map(lambda x: '<c> {} </c>'.format(x), strs))


def pack_sentences(text, key_str, values):
    split_text = re.split(HTML_REGEX, text)
    num_sent = 0
    sent_idx = 0
    toks = []
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue
        if str == '<s>':
            str = '<s {}={}>'.format(key_str, values[sent_idx])
            sent_idx += 1
            num_sent += 1
        toks.append(str)
    assert len(values) == num_sent
    return ' '.join(toks)


def sent_toks_from_html(text, convert_lower=True):
    return list(itertools.chain(*list(map(
        lambda x: x.split(' '), sents_from_html(text, convert_lower=convert_lower, extract_lr=False)))))


def is_sent_tag(str):
    return str.startswith('<s') and str.endswith('>')


def is_paragraph_tag(str):
    return str.startswith('<p') and str.endswith('>')


def paragraph_from_html(text):
    split_text = re.split(HTML_REGEX, text)
    is_tag = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
    paragraphs = []
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue
        is_paragraph_body = i > 0 and is_paragraph_tag(split_text[i - 1])
        if not is_tag[i] and is_paragraph_body:
            paragraphs.append(str)
    return paragraphs


def sents_from_html(text, convert_lower=True, extract_lr=False):
    if convert_lower:
        text = text.lower()
    split_text = re.split(HTML_REGEX, text)
    is_tag = list(map(lambda x: re.search(HTML_REGEX, x) is not None, split_text))
    sents, ranks = [], []
    for i, str in enumerate(split_text):
        str = str.strip()
        if len(str) == 0:
            continue
        is_sent_body = i > 0 and is_sent_tag(split_text[i - 1])
        if not is_tag[i] and is_sent_body:
            sents.append(str)
            if extract_lr:
                lr_val = float(split_text[i - 1].split('=')[-1].strip('<>'))
                ranks.append(lr_val)
    if extract_lr:
        return sents, np.array(ranks)
    return sents


def resolve_course(text):
    courses = list(map(lambda x: x.lstrip('<c>').rstrip('</c>').strip(), text.split('</c> <c>')))
    max_len = 0
    max_course = None
    for course in courses:
        cn = len(course)
        if cn > max_len:
            max_len = cn
            max_course = course
    return max_course
