import itertools
import pickle
from p_tqdm import p_uimap
import os
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

from rank_bm25 import BM25Plus
import pandas as pd

from preprocess.constants import out_dir
from preprocess.section_utils import resolve_course, sents_from_html


def get_target_toks(mrn):
    example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
    example_df = pd.read_csv(example_fn)
    if 'spacy_target_toks' not in example_df.columns:
        print(mrn)
        return []
    return example_df['spacy_target_toks'].apply(
        lambda x: sents_from_html(resolve_course(x), convert_lower=True)).tolist()


if __name__ == '__main__':
    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    train_df = splits_df[splits_df['split'] == 'train']
    train_mrns = train_df['mrn'].unique().tolist()
    sents_str = list(filter(None, p_uimap(get_target_toks, train_mrns)))
    sents_str_flat = list(set(list(itertools.chain(*list(itertools.chain(*sents_str))))))
    sents_fn = os.path.join(out_dir, 'train_sents.csv')
    df = pd.DataFrame(sents_str_flat, columns=['sents'])
    df.to_csv(sents_fn, index=False)

    print('Collected {} target sentences from train set.  Now indexing them for BM25...'.format(len(sents_str_flat)))
    sents_toks_flat = list(map(lambda x: x.split(' '), sents_str_flat))

    bm25 = BM25Plus(sents_toks_flat)
    out_fn = os.path.join(out_dir, 'bm25.pk')
    with open(out_fn, 'wb') as fd:
        pickle.dump(bm25, fd)