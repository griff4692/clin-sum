from collections import Counter, defaultdict
from functools import partial
import json
import math
from multiprocessing import Pool, Manager
import os
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

from lexrank import STOPWORDS
from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm

from cohort.constants import out_dir
from cohort.section_utils import sents_from_html, pack_sentences
from cohort.utils import get_mrn_status_df


class LexRank:
    def __init__(
        self,
        documents,
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True,
    ):
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words

        self.idf_score = self._calculate_idf(documents)

    def get_summary(
        self,
        sentences,
        summary_size=1,
        threshold=.1,
        fast_power_method=True,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        lex_scores = self.rank_sentences(
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )

        sorted_ix = np.argsort(lex_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        return summary

    def rank_sentences(
        self,
        sentences,
        threshold=0.1,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

        tf_scores = [
            Counter(self.tokenize_sentence(sentence)) for sentence in sentences
        ]

        similarity_matrix = self._calculate_similarity_matrix(tf_scores)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    def sentences_similarity(self, sentence_1, sentence_2):
        tf_1 = Counter(self.tokenize_sentence(sentence_1))
        tf_2 = Counter(self.tokenize_sentence(sentence_2))
        similarity = self._idf_modified_cosine([tf_1, tf_2], 0, 1)
        return similarity

    def tokenize_sentence(self, sentence):
        return list(filter(lambda x: x not in self.stopwords and not np.char.isnumeric(x), sentence.split(' ')))

    def _calculate_idf(self, documents):
        doc_number_total = len(documents)
        bags_of_words = [set(self.tokenize_sentence(doc)) for doc in documents]
        default_value = math.log(doc_number_total + 1) if self.include_new_words else 0
        idf_score = defaultdict(lambda: default_value)
        for word in set.union(*bags_of_words):
            doc_number_word = sum(1 for bag in bags_of_words if word in bag)
            idf_score[word] = math.log(doc_number_total / doc_number_word)
        return idf_score

    def _calculate_similarity_matrix(self, tf_scores):
        length = len(tf_scores)
        similarity_matrix = np.zeros([length] * 2)
        for i in range(length):
            for j in range(i, length):
                similarity = self._idf_modified_cosine(tf_scores, i, j)

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix

    def _idf_modified_cosine(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

    def _markov_matrix(self, similarity_matrix):
        row_sum = similarity_matrix.sum(axis=1, keepdims=True)

        return similarity_matrix / row_sum

    def _markov_matrix_discrete(self, similarity_matrix, threshold):
        markov_matrix = np.zeros(similarity_matrix.shape)

        for i in range(len(similarity_matrix)):
            columns = np.where(similarity_matrix[i] > threshold)[0]
            markov_matrix[i, columns] = 1 / len(columns)

        return markov_matrix


def _rank(input, lock=None, ct=None):
    mrn, account, sents = input
    sent_scores = list(lxr.rank_sentences(
        sents,
        threshold=0.1,
        fast_power_method=True,
    ))

    example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
    example_df = pd.read_csv(example_fn)
    example_df.loc[example_df.account == account, 'spacy_source_toks'] = example_df.loc[
        example_df.account == account, 'spacy_source_toks'].apply(lambda x: pack_sentences(x, 'lr', sent_scores))
    example_df.to_csv(example_fn, index=False)

    with lock:
        ct.value += 1
        if ct.value % 1000 == 0:
            print('Processed {} examples...'.format(ct.value))


if __name__ == '__main__':
    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    all_mrns = splits_df['mrn'].unique().tolist()
    train_mrns = splits_df[splits_df['split'] == 'train']['mrn'].unique().tolist()
    val_mrns = splits_df[splits_df['split'] == 'validation']['mrn'].unique().tolist()
    target_docs = []

    print('Collecting documents...')
    for i in tqdm(range(len(train_mrns))):
        example_fn = os.path.join(out_dir, 'mrn', str(train_mrns[i]), 'examples.csv')
        example_df = pd.read_csv(example_fn)
        for example in example_df.to_dict('records'):
            assert str(train_mrns[i]) == str(example['mrn'])
            target_docs.append(' '.join(sents_from_html(example['spacy_target_toks'], convert_lower=True)))

    all_source_docs = []
    full_mrns = []
    accounts = []
    for i in tqdm(range(len(all_mrns))):
        example_fn = os.path.join(out_dir, 'mrn', str(all_mrns[i]), 'examples.csv')
        example_df = pd.read_csv(example_fn)
        for example in example_df.to_dict('records'):
            all_source_docs.append(sents_from_html(example['spacy_source_toks'], convert_lower=True))
            assert str(all_mrns[i]) == str(example['mrn'])
            full_mrns.append(example['mrn'])
            accounts.append(example['account'])

    print('Precomputing target IDF...')
    stopwords = STOPWORDS['en']
    stopwords = stopwords.union([x for x in punctuation])
    lxr = LexRank(target_docs, stopwords=stopwords)
    n = len(all_source_docs)
    print('Running LexRank...')

    with Manager() as manager:
        p = Pool()
        lock = manager.Lock()
        ct = manager.Value('i', 0)
        p.map(partial(_rank, lock=lock, ct=ct), zip(full_mrns, accounts, all_source_docs))
        p.close()
        p.join()

    print('Done!')