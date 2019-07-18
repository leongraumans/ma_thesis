#!/usr/bin/python3
# Leon Graumans
#
# script for parsing scraped Tweets

import os
import re
import gzip
import json
import pickle
import pprint

hashtags = []
with open('hashtags3.txt', 'r') as f:
    for i, line in enumerate(f):
        hashtags.append(line.rstrip('\n').lower())

def remove_stuff(line):
    line = re.sub(r'@\S+','<user>', line)
    line = re.sub(r'http\S+\s?', '<url>', line)
    line = re.sub(r'www\S+\s?', '<url>', line)
    # line = re.sub(r'\#', '<hashtag>', line)

    line = re.sub(r'\|LBR\|', '', line)
    line = re.sub(r'&#x200B;', '', line)
    line = re.sub(r'\\\*', ' ', line)
    line = re.sub(r'\.(?=\S)', '. ', line)
    line = re.sub(r'\r\n', ' ', line)

    return line

def check_polarisation(text, hashtags):
    r = re.compile(r'([#])(\w+)\b')
    items = r.findall(text)
    for item in items:
        if item[1].lower() in hashtags:
            return True

def main():
    directory = 'polarised2'
    outfile = open('twitter_p2.txt', 'w')

    count = 0
    len_tokens = 0
    for root, dirs, files in os.walk(directory):
        for idx, file in enumerate(files):
            if file.endswith('.out.gz'):
                print(idx ,'- Reading ', os.path.join(directory, file), '...')
                with gzip.open(os.path.join(directory, file), 'rt', encoding='utf8') as f:
                    for i, line in enumerate(f):
                        try:
                            line = line.split('\t')[1].rstrip('\n')

                            if check_polarisation(line, hashtags):
                                line = line.rstrip()
                                line = remove_stuff(line)
                                line = line.lower()

                                count += 1
                                len_tokens += len(line.split())

                                newline = line + '\n'
                                outfile.write(newline)

                        except Exception:
                            continue

    outfile.close()
    print('Extracted {} tweets with a total of {} tokens.'.format(count, len_tokens))


if __name__ == '__main__':
    main()
