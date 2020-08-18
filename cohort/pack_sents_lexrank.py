from collections import Counter, defaultdict
import itertools
from functools import partial
import json
import math
import os
from string import punctuation
import sys
sys.path.insert(0, os.path.expanduser('~/clin-sum'))

from lexrank import STOPWORDS
from lexrank.algorithms.power_method import stationary_distribution
from lexrank.utils.text import tokenize
import numpy as np
import pandas as pd
from p_tqdm import p_uimap

from cohort.constants import out_dir
from cohort.section_utils import sents_from_html, pack_sentences
from cohort.utils import get_mrn_status_df


class LexRank:
    def __init__(
        self,
        documents=None,
        stopwords=None,
        idf_score=None,
        default=None
    ):
        self.stopwords = set() if stopwords is None else set(stopwords)
        self.default = default
        self.idf_score = idf_score or self._calculate_idf(documents)

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
        self.default = math.log(1 + doc_number_total)
        idf_score = defaultdict(lambda: self.default)
        df = defaultdict(float)
        for bow in bags_of_words:
            for word in bow:
                df[word] += 1
        for word in df:
            idf_score[word] = math.log(1 + (doc_number_total / df[word]))
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


def rank_sents(mrn):
    example_fn = os.path.join(out_dir, 'mrn', str(mrn), 'examples.csv')
    example_df = pd.read_csv(example_fn)
    if 'spacy_source_toks_packed' in example_df.columns:
        # print('Skipping MRN={}.  Already ranked and packed lex rank scores for sentences.'.format(mrn))
        return

    packed = []
    for record in example_df.to_dict('records'):
        sents = sents_from_html(record['spacy_source_toks'], convert_lower=True)
        sent_scores = list(lxr.rank_sentences(
            sents,
            threshold=0.1,
            fast_power_method=True,
        ))
        packed.append(pack_sentences(record['spacy_source_toks'], 'lr', sent_scores))
    example_df['spacy_source_toks_packed'] = packed
    example_df.to_csv(example_fn, index=False)
    return len(example_df)


if __name__ == '__main__':
    splits_df = pd.read_csv(os.path.join(out_dir, 'splits.csv'))
    all_mrns = splits_df['mrn'].unique().tolist()
    print('Running LexRank...')
    out_fn = os.path.join('..', 'data', 'idf.json')
    with open(out_fn, 'rb') as fd:
        idf = json.load(fd)

    idf_score_dd = defaultdict(lambda: idf['default'])
    print('Default IDF={}'.format(idf['default']))
    for k, v in idf['idf_score'].items():
        idf_score_dd[k] = v

    stopwords = STOPWORDS['en']
    stopwords = stopwords.union([x for x in punctuation])
    lxr = LexRank(idf_score=idf_score_dd, stopwords=stopwords, default=idf['default'])

    s = p_uimap(rank_sents, all_mrns)

    print('Processed {} examples.'.format(sum(list(s))))
    print('Done!')
