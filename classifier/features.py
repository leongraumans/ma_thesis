'''
This is a file containing features we can incorporate into the SVM models
Import this file (or individual objects from this file) as modules
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''

import re
import statistics as stats
import json
import nltk
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.sparse import csr_matrix



class Embeddings(TransformerMixin):
    '''Transformer object turning a sentence (or tweet) into a single embedding vector'''

    def __init__(self, word_embeds, word_embeds2, nrange, pte, pool='average'):
        '''
        Required input: word embeddings stored in dict structure available for look-up
        pool: sentence embeddings to be obtained either via average pooling ('average') or max pooing ('max') from word embeddings. Default is average pooling.
        '''
        self.word_embeds = word_embeds
        self.word_embeds2 = word_embeds2
        self.pool_method = pool
        self.pte = pte
        self.nrange = nrange


    def transform(self, X, **transform_params):
        '''
        Transformation function: X is list of sentence/tweet - strings in the train data. Returns list of embeddings, each embedding representing one tweet
        '''
        return [self.get_sent_embedding(sent, self.word_embeds, self.word_embeds2, self.nrange, self.pte, self.pool_method) for sent in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_sent_embedding(self, sentence, word_embeds, word_embeds2, nrange, pte, pool):
        '''
        Obtains sentence embedding representing a whole sentence / tweet
        '''

        if word_embeds2:
            l_vector = len(word_embeds['a']) * 2
        else:
            l_vector = len(word_embeds['a'])

        list_of_subword_embeddings = [
            'ft.reddit_polarised.vec',
            'ft.reddit_general.vec',
            'ft.4chan_polarised.vec',
            'ft.4chan_general.vec',
            'ft.twitter_polarised.vec',
            'crawl-300d-2M-subword.vec',
            'wiki-news-300d-1M-subword.vec',
        ]
        list_of_embeddings = []
        for word in sentence.split():
            if not word_embeds2:
                if word.lower() in word_embeds:
                    list_of_embeddings.append(word_embeds[word.lower()])

            elif word_embeds2:
                wrd =word.lower()
                if wrd in word_embeds:
                    vec1 = word_embeds[wrd]
                else:
                    vec1 = [0] * l_vector
                if wrd in word_embeds2:
                    vec2 = word_embeds2[wrd]
                else:
                    vec2 = [0] * l_vector

                vec = np.hstack((vec1, vec2))
                list_of_embeddings.append(vec)

            if nrange == '3to3':
                if word.lower() not in word_embeds and pte.split('/')[-1] in list_of_subword_embeddings:
                    # print('{} not in word_embeds, trying subwords..')
                    if len(word) > 3:
                        first_tri = word[:3]
                        if first_tri.lower() in word_embeds:
                            list_of_embeddings.append(word_embeds[first_tri.lower()])
                        last_tri = word[-3:]
                        if last_tri.lower() in word_embeds:
                            list_of_embeddings.append(word_embeds[last_tri.lower()])

            if nrange == '3to6':
                if word.lower() not in word_embeds and pte.split('/')[-1] in list_of_subword_embeddings:
                    if len(word) > 3:
                        tri_grams = [word.lower()[i:i+3] for i in range(len(word.lower())-1)]
                        for gram in tri_grams:
                            if len(gram) == 3:
                                if gram.lower() in word_embeds:
                                    list_of_embeddings.append(word_embeds[gram.lower()])

                    if len(word) > 4:
                        tri_grams = [word.lower()[i:i+4] for i in range(len(word.lower())-1)]
                        for gram in tri_grams:
                            if len(gram) == 4:
                                if gram.lower() in word_embeds:
                                    list_of_embeddings.append(word_embeds[gram.lower()])

                    if len(word) > 5:
                        tri_grams = [word.lower()[i:i+5] for i in range(len(word.lower())-1)]
                        for gram in tri_grams:
                            if len(gram) == 5:
                                if gram.lower() in word_embeds:
                                    list_of_embeddings.append(word_embeds[gram.lower()])

                    if len(word) > 6:
                        tri_grams = [word.lower()[i:i+6] for i in range(len(word.lower())-1)]
                        for gram in tri_grams:
                            if len(gram) == 6:
                                if gram.lower() in word_embeds:
                                    list_of_embeddings.append(word_embeds[gram.lower()])



        # Obtain sentence embeddings either by average or max pooling on word embeddings of the sentence
        # Option via argument 'pool'
        if pool == 'average':
            sent_embedding = [sum(col) / float(len(col)) for col in zip(*list_of_embeddings)]  # average pooling
        elif pool == 'max':
            sent_embedding = [max(col) for col in zip(*list_of_embeddings)]    # max pooling
        elif pool == 'concat':
            l_vector *= 2
            s1 = [sum(col) / float(len(col)) for col in zip(*list_of_embeddings)]  # average pooling
            s2 = [max(col) for col in zip(*list_of_embeddings)]    # max pooling
            sent_embedding = s1 + s2
            # sent_embedding = np.hstack((s1, s2))
        else:
            raise ValueError('Unknown pooling method!')

        # Below case should technically not occur
        if len(sent_embedding) != l_vector:
            # print('length embedding is not the same as vector size..')
            sent_embedding = [0] * l_vector
        return sent_embedding

class BadWords(BaseEstimator, TransformerMixin):
    '''
    Feature extractor converting each sample to number of bad words it contains normalised by its length
    Bad word list is passed in as positional argument of class object
    '''

    def __init__(self, word_file):
        ''' required input: file with list of bad words '''
        self.word_file = word_file

    def fit(self, x, y=None):
        return self

    def _get_features(self, tweet):
        '''check if twitter tokens are in a list of 'bad' words'''

        with open(self.word_file, 'r',encoding='latin-1') as fi:
            bad_list = fi.read().split(',')
        tokens = nltk.word_tokenize(tweet)
        len_tok = len(tokens)
        count = 0
        for token in tokens:
            if token in bad_list:
                count += 1
        how_bad = count/len_tok
        return {"tweet": tweet,
                "how_bad": round(how_bad,2)
                }

    def transform(self, tweets):
        '''returns a list of dictionaries, key: tweet value: results from dividing count by the number of tokens in tweet'''
        return [self._get_features(tweet) for tweet in tweets]


class TweetLength(BaseEstimator, TransformerMixin):
    """
    Transformer which turns each input sample into its length in terms of number of characters
    """

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        ''' Just get length over the whole tweet string '''

        # Desired output: matrix where axis 0 = n_sample
        values = csr_matrix([len(tweet) for tweet in X])
        return csr_matrix.transpose(values)


class FileName(BaseEstimator, TransformerMixin):
    """
    Transformer which turns each input sample into its length in terms of number of characters
    """

    def __init__(self, fntrain, fntest):
        self.fntrain = fntrain
        self.fntest = fntest

    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X):
        return self

    def transform(self, X, **transform_params):
        ''' Just get length over the whole tweet string '''
        if len(X) == len(self.fntrain):
            values = csr_matrix(self.fntrain)
        elif len(X) == len(self.fntest):
            values = csr_matrix(self.fntest)

        return csr_matrix.transpose(values)



###### Testing these classes ##################################################

if __name__ == '__main__':

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import Pipeline, FeatureUnion
    import gensim.models as gm

    tweets = []
    with open('../../Data/germeval2018.training.txt', 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 3:
                raise IndexError('Missing data for tweet "%s"' % data[0])
            tweets.append(data[0])

    print('len(tweets):', len(tweets))
    # vec_badwords = Pipeline([('badness', BadWords('lexicon.txt')), ('vec', DictVectorizer())])
    vec_lexicon = Lexicon('lexicon.txt')
    Xlex = vec_lexicon.fit_transform(tweets)
    print(type(Xlex))
    print(Xlex.shape)
    print(tweets[30:40])
    print('hello')
    print(Xlex[30:40])

    print()

    vec_len = TweetLength()
    Xlen = vec_len.fit_transform(tweets)
    print(type(Xlen))
    print(Xlen.shape)
    print(tweets[30:40])
    print(Xlen[30:40])

    '''
    embeddings = gm.KeyedVectors.load('../../Resources/hate_german.bin').wv
    print('Finished getting embeddings')
    vec_emb = Embeddings(embeddings, pool='max')
    print(vec_emb.pool_method)
    Xemb = vec_emb.fit_transform(tweets)
    # print(len(Xemb))
    # print(len(Xemb[0]))
    # print(Xemb[:2])

    count_word = CountVectorizer(ngram_range=(1,2))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,4))

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('badwords', vec_badwords),
                                ('tweetlen', TweetLength()),
                                ('word_embeds', vec_emb)])
    X = vectorizer.fit_transform(tweets)
    print(type(X))
    print(X.shape)

    '''




############### Space ####
