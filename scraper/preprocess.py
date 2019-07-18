# !/usr/bin/python3
# Leon Graumans
#
# script for cleaning & preprocessing
# using regex function of https://github.com/tkondrashov
# found on https://github.com/stanfordnlp/GloVe/issues/107

import os
import re
import sys
import emoji
import argparse

from nltk.tokenize import word_tokenize, MWETokenizer

MWET = MWETokenizer([('<', 'url', '>'), ('<', 'user', '>'), ('<', 'hashtag', '>')], separator='')


def remove_stuff(line):
    line = re.sub(r'@\S+','<user> ', line)
    line = re.sub(r'http\S+\s?', '<url> ', line)
    line = re.sub(r'www\S+\s?', '<url> ', line)
    line = re.sub(r'\#', '<hashtag> ', line)

    line = re.sub(r'\|LBR\|', '', line)
    line = re.sub(r'&#x200B;', '', line)
    line = re.sub(r'\\\*', ' ', line)
    line = re.sub(r'\.(?=\S)', '. ', line)
    line = re.sub(r'\r\n', ' ', line)

    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing before creating word embeddings')
    parser.add_argument('input', metavar='input.txt', type=str, help='Input file')
    parser.add_argument('output', metavar='output.txt', type=str, help='Output file')
    parser.add_argument('--tokenize', help='tokenize data', action='store_true')

    args = parser.parse_args()
    count = 0
    outputfile = open(args.output, 'w')
    with open(args.input, 'r') as inputfile:
        for i, line in enumerate(inputfile):

            if (i%10000==0):
                print('read {} lines..'.format(str(i)))

            line = line.rstrip()
            line = remove_stuff(line)
            line = line.lower()
            line = os.linesep.join([s for s in line.splitlines() if s])

            if not line.strip(): #check if line is empty
                continue

            if args.tokenize:
                tokens = word_tokenize(line)        # tokenize
                tokens = MWET.tokenize(tokens)      # fix <url> and <user>
                count += len(tokens)
                line = ' '.join(tokens)

            line = line + '\n'
            outputfile.write(line)
    outputfile.close()
    print('Counted {} tokens in {} tweets..'.format(count, i))
