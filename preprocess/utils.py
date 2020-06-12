import os
import re
import string
import sys

from nltk.corpus import stopwords

home_dir = os.path.expanduser('~/clin-sum/')
sys.path.insert(0, home_dir)

HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:)'
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'


def extract_section(text, matches=['history of present illness', 'hpi'], partial_match=True):
    section_regex = r'{}'.format('|'.join(matches))
    if not partial_match:
        section_regex = r'^({})$'.format(section_regex)
    sectioned_text = list(filter(lambda x: len(x.strip()) > 0, re.split(HEADER_SEARCH_REGEX, text, flags=re.M)))
    is_header_arr = list(map(lambda x: re.match(HEADER_SEARCH_REGEX, x, re.M) is not None, sectioned_text))
    is_relevant_section = list(
        map(lambda x: re.match(section_regex, x.strip(':').lower(), re.M) is not None, sectioned_text))

    relevant_section_idxs = [i for i, x in enumerate(is_relevant_section) if x]
    for i in relevant_section_idxs:
        assert is_header_arr[i]
    n = len(relevant_section_idxs)
    if n == 0:
        return None

    str = ''
    for sec_idx in relevant_section_idxs:
        str += ' <s> {} </s> <p> {} </p>'.format(sectioned_text[sec_idx].strip(), sectioned_text[sec_idx + 1].strip())
    return str.strip()


def clean_text(text):
    """
    :param text: string representing raw MIMIC note
    :return: cleaned string
    - Replace [**Patterns**] with spaces
    - Replace digits with special DIGITPARSED token
    """
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'[_*?/()]+', ' ', text)
    text = re.sub(r'\b(-)?[\d.,]+(-)?\b', ' DIGITPARSED ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def create_section_token(section):
    """
    :param section: string representing a section header as extracted from MIMIC note
    :return: string of format header=SECTIONNAME (i.e. header=HISTORYOFPRESENTILLNESS)
    """
    section = re.sub('[:\s]+', '', section).upper()
    return 'header={}'.format(section)


def create_document_token(category):
    """
    :param section: string representing a note type as provided by MIMIC
    :return: string of format document=DOCUMENTCATEGORY (i.e. document=DISCHARGESUMMARY)
    """
    category = re.sub('[:\s]+', '', category).upper()
    return 'document={}'.format(category)


def get_mimic_stopwords():
    """
    :return: set containing English stopwords plus non-numeric punctuation.
    Does not include prepositions since these are helpful for detecting nouns
    """
    other_no = set(['\'s', '`'])
    swords = set(stopwords.words('english')).union(
        set(string.punctuation)).union(other_no) - set(['%', '+', '-', '>', '<', '='])
    with open(os.path.join(home_dir, 'shared_data', 'prepositions.txt'), 'r') as fd:
        prepositions = set(map(lambda x: x.strip(), fd.readlines()))
    return swords - prepositions


def pattern_repl(matchobj):
    """
    :param matchobj: re.Match object
    :return: Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))
