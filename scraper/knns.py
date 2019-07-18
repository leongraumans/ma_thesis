#!/usr/bin/python3
# Leon Graumans
#
# script for printing 20 hand-picked k nearest neighbours for embedding file

import numpy as np
import string
from gensim.models import KeyedVectors

def open_embeddings(embeddings):

    print('loading {} embeddings..'.format(embeddings.split('/')[-1]))

    w2v = KeyedVectors.load_word2vec_format(embeddings,
                                            binary=False,
                                            unicode_errors='ignore')

    keywords = [    'woman',
                    'homosexual',
                    'black',
                    'gay',
                    'man',
                    'immigrant',
                    'immigrants',
                    'migrant',
                    'migrants',
                    'trans',
                    'gun',
                    'afroamerican',
                    'feminism',
                    'feminist',
                    'abortion',
                    'religion',
                    'god',
                    'trump',
                    'islam',
                    'muslim'
    ]

    for k in keywords:
        print('# {}'.format(k))
        i = 0
        try:
            neighbours = w2v.most_similar(k, topn=100)
            for n in neighbours:
                punc = string.punctuation + '“‘’… ”•"“'
                clean = n[0].translate(str.maketrans('', '', punc))
                if clean == k:
                    continue
                i += 1
                if i> 10:
                    break
                print(n[0], '\t', round(n[1], 4))

        except KeyError:
            print('-\n' * 10)

    print()
    print()


open_embeddings('[path to embeddings]')
