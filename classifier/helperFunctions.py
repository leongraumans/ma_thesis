'''
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''

import re
import csv
import time
import random
import codecs
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from random import shuffle
import pandas as pd
import os
import io

filename_dict = {
                    '../../public_development_en/train_en.tsv': 1,
                    '../../public_development_en/dev_en.tsv': 2,
                    '../../public_development_en/test_en.tsv': 3,
}

def loaddata(dataSet, trainPath, testPath, cls, TASK, reverse):

    IDsTrain = []
    Xtrain = []
    Ytrain = []
    fntrain = []
    IDsTest = []
    Xtest = []
    Ytest = []
    fntest = []
    if dataSet == 'WaseemHovy':
        if TASK == 'binary':
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)
        else:
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)

    elif dataSet == 'standard':
        IDsTrain,Xtrain,Ytrain,fntrain = read_corpus(trainPath,cls)
        # Also add SemEval dev-data
        IDsStandard_test,Xstandard_test,Ystandard_test,fn = read_corpus('../../public_development_en/dev_en.tsv',cls)
        for id,x,y,f in zip(IDsStandard_test,Xstandard_test,Ystandard_test,fn):
            IDsTrain.append(id)
            Xtrain.append(x)
            Ytrain.append(y)
            fntrain.append(f)
        if testPath == '../../public_development_en/test_en.tsv':
            IDsTest,Xtest,Ytest,fntest = read_corpus(testPath,cls)

    elif dataSet == 'offenseval':
        IDsTrain,Xtrain,Ytrain = read_corpus_offensevalTRAIN(trainPath,cls)
        if testPath == '../../OLIDv1.0/testset-levela.tsv':
            IDsTest,Xtest,Ytest = read_corpus_offensevalTEST(testPath,cls)

    elif dataSet == 'cross':
        ''' load train data '''

        if trainPath == '../../Full_Tweets_June2016_Dataset.csv':
            IDsTrain,Xtrain,Ytrain = read_corpus_WaseemHovy(trainPath,cls)

        elif trainPath == '../../public_development_en/train_en.tsv':
            IDsTrain,Xtrain,Ytrain,fntrain = read_corpus(trainPath,cls)
            # Also add SemEval dev-data
            IDsStandard_test,Xstandard_test,Ystandard_test,fn = read_corpus('../../public_development_en/dev_en.tsv',cls)
            for id,x,y,f in zip(IDsStandard_test,Xstandard_test,Ystandard_test,fn):
                IDsTrain.append(id)
                Xtrain.append(x)
                Ytrain.append(y)
                fntrain.append(f)
            # Also add SemEval test-data
            IDsStandard_test,Xstandard_test,Ystandard_test,fn = read_corpus('../../public_development_en/test_en.tsv',cls)
            for id,x,y,f in zip(IDsStandard_test,Xstandard_test,Ystandard_test,fn):
                IDsTrain.append(id)
                Xtrain.append(x)
                Ytrain.append(y)
                fntrain.append(f)

        # only dev for faster dev
        elif trainPath == '../../public_development_en/dev_en.tsv':
            IDsTrain,Xtrain,Ytrain,fntrain = read_corpus(trainPath,cls)

        elif trainPath == '../../OLIDv1.0/olid-training-v1.0.tsv':
            IDsTrain,Xtrain,Ytrain = read_corpus_offensevalTRAIN(trainPath,cls)
            # Also add OffensEval test-data
            IDsOther_test,Xother_test,Yother_test = read_corpus_offensevalTEST('../../OLIDv1.0/testset-levela.tsv' ,cls)
            for id,x,y in zip(IDsOther_test,Xother_test,Yother_test):
                IDsTrain.append(id)
                Xtrain.append(x)
                Ytrain.append(y)

        ''' load test data '''

        if testPath == '../../Full_Tweets_June2016_Dataset.csv':
            IDsTest,Xtest,Ytest = read_corpus_WaseemHovy(testPath,cls)

        elif testPath == '../../public_development_en/train_en.tsv':
            IDsTest,Xtest,Ytest,fntest = read_corpus(testPath,cls)
            # Also add SemEval dev-data
            IDsStandard_test,Xstandard_test,Ystandard_test,fn = read_corpus('../../public_development_en/dev_en.tsv',cls)
            for id,x,y,f in zip(IDsStandard_test,Xstandard_test,Ystandard_test,fn):
                IDsTest.append(id)
                Xtest.append(x)
                Ytest.append(y)
                fntest.append(f)
            # Also add SemEval test-data
            IDsStandard_test,Xstandard_test,Ystandard_test,fn = read_corpus('../../public_development_en/test_en.tsv',cls)
            for id,x,y,f in zip(IDsStandard_test,Xstandard_test,Ystandard_test,fn):
                IDsTest.append(id)
                Xtest.append(x)
                Ytest.append(y)
                fntest.append(f)

        elif testPath == '../../OLIDv1.0/olid-training-v1.0.tsv':
            IDsTest,Xtest,Ytest = read_corpus_offensevalTRAIN(testPath,cls)
            # Also add OffensEval test-data
            IDsOther_test,Xother_test,Yother_test = read_corpus_offensevalTEST('../../OLIDv1.0/testset-levela.tsv' ,cls)
            for id,x,y in zip(IDsOther_test,Xother_test,Yother_test):
                IDsTest.append(id)
                Xtest.append(x)
                Ytest.append(y)

    else:
        IDsTrain,Xtrain,Ytrain = read_corpus_otherSet(trainPath,cls)
        ## TODO: implement reading function for the Reddit data

    if reverse:
        tmp_id = IDsTest
        tmp_x = Xtest
        tmp_y = Ytest

        IDsTest = IDsTrain
        Xtest = Xtrain
        Ytest = Ytrain
        IDsTrain = tmp_id
        Xtrain = tmp_x
        Ytrain = tmp_y

    return IDsTrain, Xtrain, Ytrain, fntrain, IDsTest, Xtest, Ytest, fntest


def read_corpus_WaseemHovy(corpus_file,cls):
    '''Reading in data from corpus file'''
    print('Reading WaseemHovy data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='ISO-8859-1') as fi:
        for line in fi:
            data = line.strip().split(',')
            ids.append(data[0])
            if len(data)<3:
                continue
            if len(data)>3:
                tweets.append("".join(data[1:len(data) - 2]))
            else:
                tweets.append(data[1])
            if cls == 'bilstm':
                if data[len(data)-1] == 'none':
                    labels.append(0)
                else:
                    labels.append(1)
            else:
                if data[len(data)-1] == 'none':
                    labels.append('NOT')
                else:
                    labels.append('OFF')
    mapIndexPosition = list(zip(ids, tweets, labels))
    shuffle(mapIndexPosition)
    ids, tweets, labels = zip(*mapIndexPosition)

    print("read " + str(len(tweets)) + " tweets.")
    return list(ids), list(tweets), list(labels)

def read_corpus_offensevalTRAIN(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    print('Reading OffensEvalTRAIN data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
            if cls == 'bilstm':
                if data[2] == 'OFF':
                    labels.append(1)
                elif data[2] == 'NOT':
                    labels.append(0)
            else:
                if data[2] == 'OFF':
                    labels.append('OFF')
                elif data[2] == 'NOT':
                    labels.append('NOT')

    print("read " + str(len(tweets[1:])) + " tweets.")
    return ids[1:], tweets[1:], labels

def read_corpus_offensevalTEST(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    print('Reading OffensEvalTEST data...')
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 2:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
    with open('../../OLIDv1.0/labels-levela.csv', 'r', encoding='ISO-8859-1') as fi2:
        for line in fi2:
            data = line.strip().split(',')
            if len(data)<2:
                continue
            if cls == 'bilstm':
                if data[1] == 'NOT':
                    labels.append(0)
                elif data[1] == 'OFF':
                    labels.append(1)
            else:
                if data[1] == 'NOT':
                    labels.append('NOT')
                elif data[1] == 'OFF':
                    labels.append('OFF')

    print("read " + str(len(tweets[1:])) + " tweets.")
    return ids[1:], tweets[1:], labels

def read_corpus(corpus_file, cls, binary=True):
    '''Reading in data from corpus file'''
    print('Reading HatEval data...')
    ids = []
    tweets = []
    labels = []
    fn = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])
            tweets.append(data[1])
            if cls == 'bilstm':
                if data[2] == '1':
                    labels.append(1)
                elif data[2] == '0':
                    labels.append(0)
            else:
                if data[2] == '1':
                    labels.append('OFF')
                elif data[2] == '0':
                    labels.append('NOT')
            fn.append(int(filename_dict[corpus_file]))


    if corpus_file == '../../public_development_en/test_en.tsv':
        print("read " + str(len(tweets[1:])) + " tweets.")
        return ids, tweets, labels, fn
    else:
        print("read " + str(len(tweets[1:])) + " tweets.")
        return ids[1:], tweets[1:], labels, fn



 # /data/s2548798/Leon/embeddings/done/models/FastText/fastText
def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings stored as txt
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''
    print('Using embeddings: ', embedding_file)
    if embedding_file.split('/')[-1] == 'crawl-300d-2M-subword.vec':
        print('using FastText embedding..')
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
        vocab = [i for i in model.vocab.keys()]

    elif embedding_file.endswith('.txt') or embedding_file.endswith('.vec'):
        model = {}
        vocab = []
        try:
            f = open(embedding_file,'r')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    float(values[1])
                except ValueError:
                    continue
                coefs = np.asarray(values[1:], dtype='float')
                model[word] = coefs
                vocab.append(word)
        except UnicodeDecodeError:
            f = open(embedding_file,'rb')
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    float(values[1])
                except ValueError:
                    continue
                coefs = np.asarray(values[1:], dtype='float')
                model[word] = coefs
                vocab.append(word)
        f.close()

    print ("Done.",len(vocab)," words loaded!")
    print('Type model: {}'.format(type(model)))
    return model, vocab




def clean_samples(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    new_samples = []
    for line in samples:

        line = re.sub(r'@\S+','<user>', line)
        line = re.sub(r'http\S+\s?', '<url>', line)
        line = re.sub(r'www\S+\s?', '<url>', line)
        line = re.sub(r'\#', '<hashtag>', line)

        line = re.sub(r'\|LBR\|', '', line)
        line = re.sub(r'&#x200B;', '', line)
        line = re.sub(r'\\\*', ' ', line)
        line = re.sub(r'\.(?=\S)', '. ', line)
        line = re.sub(r'\r\n', ' ', line)
        new_samples.append(line)

    return new_samples

def clean_samples_ruby(samples):
    tmpname = 'tmpdir/tmp_' + str(time.time()) + '.txt'
    with open(tmpname, 'w') as tmp_file:
        for line in samples:
            tmp_file.write(line + '\n')

    command_tmp = 'ruby -n preprocess-twitter.rb ' + tmpname
    clean = os.popen(command_tmp).read().split('\n')

    new_samples = []
    for line in clean[:-1]:
        new_samples.append(line)
    return new_samples

def load_offense_words(path):
    ow = []
    f = open(path, "r")
    for line in f:
        ow.append(line[:-1])
    return ow


def parse_classification_report(report):
    reportlist = report.strip().split('\n')
    newlist = []
    for i in reportlist:
        if i != '':
            i = i.split()[-4:-1]
            newlist.append(i)
    newlist = newlist[1:-1]
    NOT_prec = float(newlist[0][0])
    NOT_rec = float(newlist[0][1])
    NOT_f1 = float(newlist[0][2])

    OFF_prec = float(newlist[1][0])
    OFF_rec = float(newlist[1][1])
    OFF_f1 = float(newlist[1][2])

    return NOT_prec, NOT_rec, NOT_f1, OFF_prec, OFF_rec, OFF_f1


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    # top_feats = [features[row]]
    # print(top_feats)
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''
    labs = sorted(set(Ygold + Yguess.tolist()))
    """ print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:") """

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array



    # printing out precision, recall, f-score for each class in easily readable way
    PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
    """ print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))

    print('-'*50)
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print() """

    return [PRFS, labs]


def mean(list):
    return sum(list)/len(list)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
