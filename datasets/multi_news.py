from collections import Counter
import itertools
import os
import pickle
import re

import nlp
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from datasets.tokenize_utils import clean

import tqdm

import nltk

DOC_DELIM = '|||||'
HOME_DIR = os.path.expanduser('/Users/nihaar/Documents/Spring20/research/abstractive-summarization/code/clin-sum/')
# Special characters

PAD = '<pad>'
UNK = '<unk>'
SENTENCE = '<s>'
PG = '<p>'
START = '<start>'
END = '<end>'


SPECIAL_CHARS = [PAD, UNK, SENTENCE, PG, START, END]


class Example:
    def __init__(self, doc_toks, summary_toks):
        self.doc_toks = doc_toks
        self.summary_toks = summary_toks

        self.enc_ids = None
        self.dec_input_ids = None
        self.dec_target_ids = None
        self.doc_oov = None

    def to_ids(self, vocab):
        start_idx = vocab.stoi['<start>']
        end_idx = min(vocab.stoi['<end>'],75)
        # end_idx = 75
        # Convert to list of ids
        doc_ids_w_unk, doc_ids_extended, doc_oov = doc_to_ids(self.doc_toks, vocab)

        # Get a version of the reference summary where in-article OOVs are represented by their temporary article OOV id
        sum_ids_w_unk, sum_ids_extended = summary_to_ids(self.summary_toks, vocab, doc_oov)
        dec_input_ids, dec_target_ids = get_dec_inp_targ_seqs(sum_ids_w_unk, sum_ids_extended, 50, start_idx, end_idx)

        self.doc_oov = doc_oov
        self.enc_input_ids = doc_ids_w_unk
        self.enc_target_ids = doc_ids_extended
        self.dec_input_ids = dec_input_ids
        self.dec_target_ids = dec_target_ids

    def tensorize(self):
        pass


def remove_special(arr):
    return list(filter(lambda tok: tok not in SPECIAL_CHARS, arr))


def remove_empty(arr):
    return list(filter(lambda x: len(x) > 0, arr))


def ids_to_toks(toks, vocab):
    return list(map(vocab.itos.__getitem__, toks))


def toks_to_ids(toks, vocab):
    return [vocab.stoi[tok] if tok in vocab.stoi else vocab.stoi[vocab.UNK] for tok in toks]


def summary_to_ids(summary_toks, vocab, doc_oovs):
    """Map the summary words to their ids. In-article OOVs are mapped to their temporary OOV numbers.
    """
    ids_w_unk, ids_extended = [], []
    unk_id = vocab.stoi.get('<unk>')
    V = len(vocab.stoi)
    for tok in summary_toks:
        id = vocab.stoi.get(tok, unk_id)
        ids_w_unk.append(id)
        if id == unk_id:  # If w is an OOV word
          if tok in doc_oovs:  # If w is an in-article OOV
            vocab_idx = V + doc_oovs.index(tok)  # Map to its temporary article OOV number
            ids_extended.append(vocab_idx)
          else:  # If w is an out-of-article OOV
            ids_extended.append(unk_id)  # Map to the UNK token id
        else:
          ids_extended.append(id)
    return ids_w_unk, ids_extended


def doc_to_ids(doc_toks, vocab):
    """Map the document words to their ids. Also return a list of OOVs in the article.
    """
    ids_w_unk, ids_extended = [], []
    oovs = []
    unk_id = vocab.stoi['<unk>']
    V = len(vocab.stoi)
    for doc in doc_toks:
        id_w_unk_row, id_extended_row = [], []
        for tok in doc:
            i = vocab.stoi.get(tok, unk_id)
            id_w_unk_row.append(i)
            if i == unk_id:  # If w is OOV
              if tok not in oovs:  # Add to list of OOVs
                oovs.append(tok)
              oov_num = oovs.index(tok)  # This is 0 for the first article OOV, 1 for the second article OOV...
              id_extended_row.append(V + oov_num)  # This is e.g. 50000 for the first OOV, 50001 for the second...
            else:
              id_extended_row.append(i)
        ids_extended.append(id_extended_row)
        ids_w_unk.append(id_w_unk_row)
    return ids_w_unk, ids_extended, oovs


def _word_tokenize(str):
    toks = remove_empty(word_tokenize(str))
    if len(toks) > 0:
        if toks[-1] == '.' or len(toks[-1]) == 0:
            toks = toks[:-1]
        else:
            toks[-1] = toks[-1].strip('.')
    return list(map(lambda x: '<p>' if x == 'paragraphdelim' else x, toks)) + ['<s>']


def tokenize(str):
    docs = clean(str).lower()
    docs = remove_empty(docs.split(DOC_DELIM))
    docs_p_delim = list(map(lambda doc: re.sub(r'\s+\n\s+\n\s+', ' paragraphdelim ', doc), docs))
    doc_sents = list(remove_empty(map(sent_tokenize, docs_p_delim)))
    toks = list(map(lambda sents: list(itertools.chain(*list(map(_word_tokenize, sents))))[:-1], doc_sents))
    return toks


def get_dec_inp_targ_seqs(ids_w_unk, ids_w_extended, max_len, start_id, stop_id):
    """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the
    target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len.
    The input sequence must start with the start_id and the target sequence must end with the stop_id
    (but not if it's been truncated).
    Args:
    sequence: List of ids (integers)
    max_len: integer
    start_id: integer
    stop_id: integer
    Returns:
    inp: sequence length <=max_len starting with start_id
    target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + ids_w_unk[:]
    target = ids_w_extended[:]
    if len(inp) > max_len:  # truncate
        inp = inp[:max_len]
        target = target[:max_len]  # no end_token
    else:  # no truncation
        target.append(stop_id)  # end token
    assert len(inp) == len(target)
    return inp, target


class MultiNewsDataset(Dataset):
    def __init__(self, type, data):
        self.type = type
        self.vocab = None

        processed_data = list(map(self._preprocess_example, data))

        # TODO[deprecate]: If you want to debug, uncomment this
        processed_data = []
        for i, example in enumerate(data):
            processed_data.append(self._preprocess_example(example))
            if i > 10: # presumably: to limit the number of examples in the training data
                break
        self.examples = list(map(lambda x: Example(x[0], x[1]), processed_data))

    def _preprocess_example(self, example):
        return tokenize(example['document']), tokenize(example['summary'])[0]

    def add_ids_to_examples(self):
        for example in self.examples:
            example.to_ids(self.vocab)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def load():
    preprocess_fn = os.path.join(HOME_DIR, 'datasets/preprocessed/multi_news.pk')
    if os.path.exists(preprocess_fn):
        processed_dataset = pickle.load(open(preprocess_fn, 'rb'))
    else:
        full_dataset = nlp.load_dataset('multi_news')
        print("Downlaoded the data")
        processed_dataset = {k: MultiNewsDataset(k, v) for k, v in full_dataset.items()}

        toks = []
        for k in ['train', 'validation']:
            doc_tok_chain = map(lambda example: example.doc_toks, processed_dataset[k])
            summary_tok_chain = map(lambda example: example.summary_toks, processed_dataset[k])
            toks += list(itertools.chain(*list(itertools.chain(*doc_tok_chain))))
            toks += list(itertools.chain(*summary_tok_chain))
        toks = remove_special(toks)
        print("Obtained the tokens of size",len(toks))
        # preprocess_fn = '/Users/nihaar/Documents/Spring20/research/abstractive-summarization/code/clin-sum/datasets/preprocessed/multi_news2.pk'
        vocab = Vocab(Counter(toks), specials=SPECIAL_CHARS, specials_first=True, min_freq=5)
        for k in processed_dataset:
            processed_dataset[k].vocab = vocab
            processed_dataset[k].add_ids_to_examples()

        with open(preprocess_fn, 'wb') as fd:
            pickle.dump(processed_dataset, fd)

    return processed_dataset


# load()