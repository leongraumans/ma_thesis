'''
SVM system for detecting abusive language in multiple data sets
original by Caselli et al: https://github.com/malvinanissim/germeval-rug
'''
#### PARAMS #############################################
# source = 'Twitter', 'Reddit' or '' chooses tokenizer or none
# ftr = 'ngram', 'embeddings' or 'embeddings+ngram'
# cls = 'bilstm' or ''
# clean = 'none', 'std' or 'ruby'
# evlt = 'cv10' or 'traintest'

# dataSet = 'standard'      #HatEval
# dataSet = 'WaseemHovy'    #WaseemHovy
# dataSet = 'offenseval'    #OffensEval
# dataSet = 'cross'         #Cross testing

# modelh5 = bilstm model
# tknzr = bilstm tokenizer

# trainPath = path to trainfile
# testPath = path to testfile or empty
# path_to_embs = path to embedding

# evlt = 'cv10' or 'traintest'
#########################################################

import helperFunctions
import transformers
import features
import argparse
import re
import stop_words
import json
import pickle
import os
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.tokenize import TweetTokenizer, word_tokenize, MWETokenizer

seed = 1337
np.random.seed(seed)

MWET = MWETokenizer([   ('<', 'url', '>'),
                        ('<', 'user', '>'),
                        ('<', 'smile', '>'),
                        ('<', 'lolface', '>'),
                        ('<', 'sadface', '>'),
                        ('<', 'neutralface', '>'),
                        ('<', 'heart', '>'),
                        ('<', 'number', '>'),
                        ('<', 'hashtag', '>'),
                        ('<', 'allcaps', '>'),
                        ('<', 'repeat', '>'),
                        ('<', 'elong', '>'),
                    ], separator='')

def ntlktokenizer(x):
    tokens = word_tokenize(x)           # tokenize
    tokens = MWET.tokenize(tokens)      # fix <url> and <user> etc.

    return ' '.join(tokens)


def check_coverage(x, embeddings):
    words = []
    for tweet in x:
        tokens = word_tokenize(tweet)           # tokenize
        tokens = MWET.tokenize(tokens)          # fix <url> and <user> etc.
        for x in tokens:
            words.append(x)
    vocab2 = list(set(words))

    yes = 0
    no = 0
    for word in vocab2:
        if word in embeddings.keys():
            yes += 1
        else:
            no += 1
    print('{} words in embedding, {} are NOT in embedding..'.format(yes, no))

def main():
# example:
# python3 SVM_original.py -src '' -ftr 'embeddings' -cls 'bilstm' -ds 'WaseemHovy' -mh5 'models/CVWaseem_reddit_general_ruby_weights-improvement-{epoch:02d}-{loss:.2f}.h5' -tknzr 'models/CVWaseem_tokenizer.pickle' -trnp '../../Full_Tweets_June2016_Dataset.csv' -tstp '' -pte '../../embeddings/reddit_general_ruby.txt' -evlt '' -cln 'none'

    parser = argparse.ArgumentParser(description='ALW')
    parser.add_argument('-src', type=str, help='source')
    parser.add_argument('-ftr', type=str, help='ftr')
    parser.add_argument('-cls', type=str, help='cls')
    parser.add_argument('-ds', type=str, help='dataSet')
    parser.add_argument('-mh5', type=str, help='modelh5')
    parser.add_argument('-tknzr', type=str, help='tknzr')
    parser.add_argument('-trnp', type=str, help='trainPath')
    parser.add_argument('-tstp', type=str, help='testPath')
    parser.add_argument('-pte', type=str, help='path_to_embs')
    parser.add_argument('-pte2', type=str, help='path_to_embs2')
    parser.add_argument('-evlt', type=str, help='evlt')
    parser.add_argument('-cln', type=str, help='clean')
    parser.add_argument('-pool', type=str, default='max', help='pool max/average/concat')
    parser.add_argument('-eps', type=int, help='epochs')
    parser.add_argument('-ptc', type=int, help='patience')
    parser.add_argument('-vb', type=int, help='verbose')
    parser.add_argument('-bs', type=int, help='batch_size', default=64)
    parser.add_argument('-nrange', type=str, help='fasttext nrange', default='1')
    parser.add_argument('-lstmTrn', type=helperFunctions.str2bool, help='bilstm training True/False')
    parser.add_argument('-lstmOp', type=helperFunctions.str2bool, help='bilstm output True/False')
    parser.add_argument('-lstmTd', type=helperFunctions.str2bool, help='bilstm traindev True/False')
    parser.add_argument('-lstmCV', type=helperFunctions.str2bool, help='bilstm cv True/False')
    parser.add_argument('-prb', type=helperFunctions.str2bool, help='yguess_output/probabilities True/False')
    parser.add_argument('-cnct', type=helperFunctions.str2bool, help='concat')
    parser.add_argument('--tokenize', help='tokenize data', action='store_true')
    parser.add_argument('--reverse', help='tokenize data', action='store_true')
    parser.add_argument('--stackembeds', help='tokenize data', action='store_true')

    args = parser.parse_args()

    source = args.src
    ftr = args.ftr
    cls = args.cls
    dataSet = args.ds
    modelh5 = args.mh5
    tknzr = args.tknzr
    trainPath = args.trnp
    testPath = args.tstp
    path_to_embs = args.pte
    path_to_embs2 = args.pte2
    evlt = args.evlt
    clean = args.cln
    lstmTraining = args.lstmTrn
    lstmOutput = args.lstmOp
    lstmTrainDev = args.lstmTd
    lstmCV = args.lstmCV
    lstmEps = args.eps
    lstmPtc = args.ptc
    reverse = args.reverse
    vb = args.vb
    bs = args.bs
    prob = args.prb
    concat = args.cnct
    pool = args.pool
    nrange = args.nrange
    stack_embeds = args.stackembeds

    TASK = 'binary'
    #TASK = 'multi'

    '''
    Preparing data
    '''

    print('Reading in ' + source + ' training data using ' + dataSet + 'dataset...')

    IDsTrain, Xtrain, Ytrain, FNtrain, IDsTest, Xtest, Ytest, FNtest = helperFunctions.loaddata(dataSet, trainPath, testPath, cls, TASK, reverse)

    print('Done reading in data...')

    '''
    Preprocessing / cleaning
    '''

    if clean == 'std':
        Xtrain = helperFunctions.clean_samples(Xtrain)
        if testPath != '':
            Xtest = helperFunctions.clean_samples(Xtest)
    if clean == 'ruby':
        Xtrain = helperFunctions.clean_samples_ruby(Xtrain)
        if testPath != '':
            Xtest = helperFunctions.clean_samples_ruby(Xtest)

    print(len(Xtrain), 'training samples after cleaning!')
    if Xtest:
        print(len(Xtest), 'testing samples after cleaning!')

    '''
    Tokenizing
    '''

    if args.tokenize:
        print('Tokenizing data...')
        Xtrain_tok = []
        Xtest_tok = []
        for line in Xtrain:
            tokens = word_tokenize(line)        # tokenize
            tokens = MWET.tokenize(tokens)      # fix <url> and <user>
            line = ' '.join(tokens)
            Xtrain_tok.append(line)
        for line in Xtest:
            tokens = word_tokenize(line)        # tokenize
            tokens = MWET.tokenize(tokens)      # fix <url> and <user>
            line = ' '.join(tokens)
            Xtest_tok.append(line)
        Xtrain = Xtrain_tok
        Xtest = Xtest_tok

    '''
    Calculating class ratio
    '''

    if cls == 'bilstm':
        offensiveRatio = Ytrain.count(1)/len(Ytrain)
        nonOffensiveRatio = Ytrain.count(0)/len(Ytrain)
    else:
        offensiveRatio = Ytrain.count('OFF')/len(Ytrain)
        nonOffensiveRatio = Ytrain.count('NOT')/len(Ytrain)

    print('OFF {}'.format(offensiveRatio))
    print('NOT {}'.format(nonOffensiveRatio))


    '''
    Preparing vectorizer and classifier
    '''

    print('Preparing tools (vectorizer, classifier) ...')

    if source == 'Twitter':
        tokenizer = TweetTokenizer().tokenize
    elif source == 'Reddit':
        tokenizer = ntlktokenizer
    else:
        tokenizer = None


    embeddings = {}
    if ftr == 'ngram':
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
        vectorizer = FeatureUnion([ ('word', count_word),
                                    ('char', count_char)])


    elif ftr =='custom':
        vectorizer = FeatureUnion([
                                    # ('tweet_length', features.TweetLength()),
                                    ('file_name', features.FileName(FNtrain, FNtest))
        ])

    elif ftr == 'embeddings':
        print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        print('Done')

        if stack_embeds:
            print('Stacking embeddings..')
            embeddings2, vocab2 = helperFunctions.load_embeddings(path_to_embs2)
            vectorizer = FeatureUnion([
                                        ('word_embeds', features.Embeddings(embeddings, embeddings2, nrange, path_to_embs, pool=pool)),
                                        # ('tweet_length', features.TweetLength()),
                                        # ('file_name', features.FileName(FNtrain, FNtest))
            ])
        else:
            embeddings2 = {}
            vectorizer = FeatureUnion([
                                        ('word_embeds', features.Embeddings(embeddings, embeddings2,nrange, path_to_embs, pool=pool)),
                                        # ('tweet_length', features.TweetLength()),
                                        # ('file_name', features.FileName(FNtrain, FNtest))
            ])

        # print('\n\nCreating vocab for Xtrain...')
        # check_coverage(Xtrain, embeddings)
        # if testPath != '':
        #     print('Creating vocab for Xtest...')
        #     check_coverage(Xtest, embeddings)
        #
        # exit()


    elif ftr == 'embeddings+ngram':
        count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('en'), tokenizer=tokenizer)
        count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))
        # path_to_embs = 'embeddings/model_reset_random.bin'
        print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
        embeddings, vocab = helperFunctions.load_embeddings(path_to_embs)
        print('Done')
        vectorizer = FeatureUnion([ ('word', count_word),
                                    ('char', count_char),
                                    ('word_embeds', features.Embeddings(embeddings, nrange, path_to_embs, pool=pool))])

    if cls == 'bilstm':
        from BiLSTM import biLSTM
        if lstmTrainDev:
            print('Splitting train data into 66% train + 33% test..')
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain, Ytrain, test_size=0.33, random_state=seed)
        print('Train labels', set(Ytrain), len(Ytrain))
        print('Test labels', set(Ytest), len(Ytest))
        Ytest, Yguess = biLSTM(Xtrain, Ytrain, Xtest, Ytest, lstmTraining, lstmOutput, embeddings, embeddings2, tknzr, modelh5, lstmCV, lstmEps, lstmPtc, dataSet, vb, bs, prob)


        for text, gold, guess in zip(Xtest, Ytest, Yguess):
            if gold != guess:
                print('gold: {}, guess: {}, tweet:  {}'.format(gold, guess, text))



    # Set up SVM classifier with unbalanced class weights
    if TASK == 'binary' and cls != 'bilstm':
        cl_weights_binary = {'NOT':1/nonOffensiveRatio, 'OFF':1/offensiveRatio}
        clf = SVC(kernel='linear', probability=True, class_weight=cl_weights_binary, random_state=seed)
    elif TASK == 'multi':
        cl_weights_multi = {'OTHER':0.5,
                            'ABUSE':3,
                            'INSULT':3,
                            'PROFANITY':4}
        clf = SVC(kernel='linear', class_weight=cl_weights_multi, probability=True, random_state=seed)

    if cls != 'bilstm':
        classifier = Pipeline([
                                ('vectorize', vectorizer),
                                ('classify', clf)])


    '''
    Actual training and predicting:
    '''

    if evlt == 'cv10':
        print('10-fold cross validation results:')
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)

        accuracy = 0
        precision = 0
        recall = 0
        fscore = 0

        NOT_prec = 0
        NOT_rec = 0
        NOT_f1 = 0
        OFF_prec = 0
        OFF_rec = 0
        OFF_f1 = 0
        for train_index, test_index in kfold.split(Xtrain, Ytrain):
            X_train, X_test = Xtrain[train_index], Xtrain[test_index]
            Y_train, Y_test = Ytrain[train_index], Ytrain[test_index]

            classifier.fit(X_train,Y_train)
            Yguess = classifier.predict(X_test)

            accuracy += accuracy_score(Y_test, Yguess)
            precision += precision_score(Y_test, Yguess, average='macro')
            recall += recall_score(Y_test, Yguess, average='macro')
            fscore += f1_score(Y_test, Yguess, average='macro')
            report = classification_report(Y_test, Yguess)

            notp, notr, notf1, offp, offr, off1 = helperFunctions.parse_classification_report(report)

            NOT_prec += notp
            NOT_rec += notr
            NOT_f1 += notf1
            OFF_prec += offp
            OFF_rec += offr
            OFF_f1 += off1

        print("NOTprec  {}  NOTrec  {}  NOTf1  {}".format(NOT_prec/10, NOT_rec/10, NOT_f1/10))
        print("OFFprec  {}  OFFrec  {}  OFFf1  {}".format(OFF_prec/10, OFF_rec/10, OFF_f1/10))

        print('accuracy_score: {}'.format(accuracy / 10))
        print('precision_score: {}'.format(precision / 10))
        print('recall_score: {}'.format(recall / 10))
        print('f1_score: {}'.format(fscore / 10))

        print('\n\nDone, used the following parameters:')
        print('train: {}'.format(trainPath))
        if ftr != 'ngram':
            print('embed: {}'.format(path_to_embs))
        print('feats: {}'.format(ftr))
        print('prepr: {}'.format(clean))
        print('sourc: {} - datas: {}'.format(source, dataSet))
    elif evlt == 'traintest':
        if cls != 'bilstm':
            classifier.fit(list(Xtrain),Ytrain)
            Yguess = classifier.predict(Xtest)
        print('train test results:')
        print(accuracy_score(Ytest, Yguess))
        print(set(Ytest), set(Yguess))
        print(precision_recall_fscore_support(Ytest, Yguess, average='macro'))
        report = classification_report(Ytest, Yguess)
        print(report)
        notp, notr, notf1, offp, offr, off1 = helperFunctions.parse_classification_report(report)
        print('{}\t{}\t{}\t{}\t{}\t{}'.format(str(notp).replace('.', ','), str(notr).replace('.', ','), str(notf1).replace('.', ','), str(offp).replace('.', ','), str(offr).replace('.', ','), str(off1).replace('.', ',')))
        print('\n\nDone, used the following parameters:')
        print('train: {}'.format(trainPath))
        print('tests: {}'.format(testPath))
        if ftr != 'ngram':
            print('embed: {}'.format(path_to_embs))
        print('feats: {}'.format(ftr))
        print('prepr: {}'.format(clean))
        print('sourc: {} - datas: {}'.format(source, dataSet))

    if prob:
        with open('probas_SVC_' + dataSet + '_' + trainPath.split("/")[-1] + '_' + ftr + '_' + path_to_embs.split("/")[-1] + '_concat=' + str(concat) + '.txt', 'w+') as yguess_output:
            for i in classifier.predict_proba(Xtest):
                yguess_output.write('%s\n' % i[1])

if __name__ == '__main__':
    main()
