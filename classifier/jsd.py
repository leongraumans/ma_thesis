'''
Script for calculating the jensen shannon divergence between two datasets
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''

import re, math, collections
from collections import Counter
import numpy as np
import scipy
from scipy import stats
import sys
from nltk.corpus import stopwords

def tokenize(_str):
    tokens = collections.defaultdict(lambda: 0.)
    for m in re.finditer(r"(\w+)", _str, re.UNICODE):
        m = m.group(1).lower()
        if len(m) < 2:
            continue
        if m in stopwords.words('english'):
            continue
        tokens[m] += 1
    return tokens
#end of tokenize

def kldiv(_s, _t):
    if (len(_s) == 0):
        return 1e33

    if (len(_t) == 0):
        return 1e33

    ssum = 0. + sum(_s.values())
    slen = len(_s)

    tsum = 0. + sum(_t.values())
    tlen = len(_t)

    vocabdiff = set(_s.keys()).difference(set(_t.keys()))
    lenvocabdiff = len(vocabdiff)

    """ epsilon """
    epsilon = min(min(_s.values())/ssum, min(_t.values())/tsum) * 0.001

    """ gamma """
    gamma = 1 - lenvocabdiff * epsilon

    """ Check if distribution probabilities sum to 1"""
    sc = sum([v/ssum for v in _s.itervalues()])
    st = sum([v/tsum for v in _t.itervalues()])

    vocab = Counter(_s) + Counter(_t)
    ps = []
    pt = []
    for t, v in vocab.iteritems():
        if t in _s:
            pts = gamma * (_s[t] / ssum)
        else:
            pts = epsilon

        if t in _t:
            ptt = gamma * (_t[t] / tsum)
        else:
            ptt = epsilon

        ps.append(pts)
        pt.append(ptt)

    return ps, pt


def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    """from https://github.com/sebastianruder/learn-to-select-data/blob/master/similarity.py """
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim



if __name__ == '__main__':

    """
    Usage python2 jsd.py [INPUT_DOC1] [INPUT_DOC2]

    It takes in input 2 .txt file (i.e. all the Xs of a dataset) and gives
    a score as output.     The script tokenize the data and remove stopwords.
    The input data must contains token strings not numbers.
    """

    d1 = sys.argv[1]
    d2 = sys.argv[2]

    in1 = d1.split("/")[-1]
    in2 = d2.split("/")[-1]

    f1 = ''
    with open(d1, 'r') as fi:
        for line in fi:
            f1 += line

    f2 = ''
    with open(d2, 'r') as fi:
        for line in fi:
            f2 += line

    kldiv(tokenize(d1), tokenize(d2))

    d1_ = kldiv(tokenize(f1), tokenize(f2))[0]
    d2_ = kldiv(tokenize(f1), tokenize(f2))[1]

    repr1 = np.asarray(d1_)
    repr2 = np.asarray(d2_)

    output = open("j-s_documentLevel-trainRED-testTE.txt", 'a')
    output.writelines(in1 + "\t" + in2 + "\t" + str(jensen_shannon_divergence(repr1,repr2)) + "\n")
    output.close()

    print jensen_shannon_divergence(repr1,repr2)
