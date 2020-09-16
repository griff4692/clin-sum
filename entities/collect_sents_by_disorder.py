from collections import defaultdict
import itertools
import os
import re
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

import pandas as pd
import spacy

from preprocess.constants import out_dir
from preprocess.section_utils import paragraph_from_html
from preprocess.tokenize_mrns import sent_segment

FINDING_TUI = 'T033'
DISORDER_CUI = 'C0024198'


def get_paragraph_mentions(text, mentions, cui):
    output = []
    mention_regex = r'|'.join([re.escape(m) for m in mentions])
    sents = sent_segment(text, sentencizer=spacy_nlp)
    for sent in sents:
        match = re.search(mention_regex, sent)
        if match is not None:
            s, e = match.span()
            packed_sent = sent[:s] + ' <' + cui + '> ' + sent[s:e] + ' </' + cui + '> ' + sent[e:]
            output.append(packed_sent)

    return output


if __name__ == '__main__':
    print('Loading scispacy')
    spacy_nlp = spacy.load('en_core_sci_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

    lymes_fn = os.path.join(out_dir, 'entity', 'lymes_entities.csv')
    if os.path.exists(lymes_fn):
        print('Loading cached file {}'.format(lymes_fn))
        disorder_df = pd.read_csv(lymes_fn)
    else:
        print('Loading entities...')
        ent_df = pd.read_csv(os.path.join(out_dir, 'entity', 'full_entities.csv'))
        # valid_df = ent_df[((ent_df['target_count'] > 0) & (ent_df['source_count'] > 0))]
        # disorder_df = valid_df[((valid_df['sem_group'] == 'Disorders') & (valid_df['tui'] != FINDING_TUI))]
        disorder_df = ent_df[ent_df['cui'] == DISORDER_CUI]
        disorder_df.to_csv(lymes_fn, index=False)

    accounts = disorder_df['account'].unique().tolist()
    num_examples = len(accounts)
    output_df = defaultdict(list)
    for account in accounts:
        account_df = disorder_df[disorder_df['account'] == account]
        num_source = account_df['is_source'].sum()
        num_target = account_df['is_target'].sum()
        if num_source == 0 or num_target == 0:
            continue

        source_text_mentions = account_df[account_df['is_source']]['extracted_text']
        target_text_mentions = account_df[account_df['is_target']]['extracted_text']

        mrn = account_df['mrn'].iloc[0]
        example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
        mrn_df = pd.read_csv(example_fn)
        account_df = mrn_df[mrn_df['account'] == account]
        assert len(account_df) == 1
        example = account_df.iloc[0]
        source_paragraphs = paragraph_from_html(example['source_str'])
        target_paragraphs = paragraph_from_html(example['target_str'])
        assert len(source_paragraphs) >= 1 and len(target_paragraphs) >= 1
        source_packed_sents = list(itertools.chain(
            *[get_paragraph_mentions(sp, source_text_mentions, DISORDER_CUI) for sp in source_paragraphs]))
        target_packed_sents = list(itertools.chain(
            *[get_paragraph_mentions(tp, target_text_mentions, DISORDER_CUI) for tp in target_paragraphs]))

        output_df['mrn'].append(mrn)
        output_df['account'].append(account)
        output_df['source_packed_sents'].append(' <s> '.join(source_packed_sents))
        output_df['target_packed_sents'].append(' <s> '.join(target_packed_sents))

    output_df = pd.DataFrame(output_df)
    out_fn = os.path.join(out_dir, 'entity', 'lymes_examples.csv')
    output_df.to_csv(out_fn, index=False)
    print('{} valid examples out of {}'.format(len(output_df), len(accounts)))