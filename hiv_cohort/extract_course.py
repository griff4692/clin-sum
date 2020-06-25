import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# CONSTANTS
MAX_HEADER_LEN = 50
# allowable number of extra chars on either side of
# {'course', 'fen', 'gi', 'id', etc.}
# to count as acceptable hospital course (sub)header
MAX_SYSTEMS_COURSE_HEADER_LEN = 15
MIN_TARGET_LEN = 25  # min number of allowable characters in body of hospital course section

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
    r'maintenance', r'measure', r'\bnote\b', r'\bvital', r'(on|at)? ?discharge', r'path(ology)?'
]


def same_text(a, b):
    return re.sub(r'\W+', '', a.lower()) == re.sub(r'\W+', '', b.lower())


def clean(text):
    cleaned = []
    for line in text.split('\n'):
        line_strip = ''.join([x if ord(x) < 128 else '' for x in line]).strip()
        line_strip = re.sub(r'[-_.<\s]{10,}', ' ', line_strip)
        line_strip = re.sub(r'\s+\[retrieved for.*\]', '', line_strip)
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

    # To add
    # either it does not start with a section header or it is an allowable section header
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


def extract_hospital_course(text):
    sectioned_text = re.split(HEADER_REGEX, text, flags=re.M)
    sectioned_text = [x.lstrip() for x in sectioned_text if len(x.lstrip()) > 0]
    is_header_arr = list(map(lambda x: re.match(HEADER_REGEX, x, re.M) is not None, sectioned_text))
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
    return ' '.join(map(lambda x: '<s> {} </s>'.format(x), strs))


def main():
    df = pd.read_csv('/nlp/projects/clinsum/dsum_sample.csv')
    df['is_dsum'] = df['note_type'].apply(is_dsum)
    df = df[df['is_dsum']]
    courses = []
    records = df.to_dict('records')
    n = len(records)
    for i in tqdm(range(n)):
        cleaned_text = clean(records[i]['text'])
        courses.append(extract_hospital_course(cleaned_text))

    df['course'] = courses
    df.to_csv('/nlp/projects/clinsum/dsum_sample_annotated.csv', index=False)


def is_dsum(notetype):
    if notetype is None:
        return False
    ntl = notetype.lower()
    return ('discharge summary' in ntl or 'discharge note' in ntl) and not 'nurse' in ntl and not 'nursing' in ntl


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'run':
        main()
    else:
        df = pd.read_csv('/nlp/projects/clinsum/dsum_sample_annotated.csv')
        df['is_dsum'] = df['note_type'].apply(is_dsum)
        df = df[df['is_dsum']]
        records = df.to_dict('records')
        i = 0 if len(sys.argv) == 2 else int(sys.argv[2])
        record = records[i]
        print(record['text'])
        print('\n\n' + '_' * 100 + '\n\n')
        print(record['note_type'], ', ', record['title'])
        print('\n\n' + '_' * 100 + '\n\n')
        tl = record['text'].lower()
        course_in_src = 'hospital course' in tl or 'course:' in tl
        if course_in_src:
            print('Course in Source')
        else:
            print('Course NOT in source')
        print('\n\n' + '_' * 100 + '\n\n')
        print(record['course'])
