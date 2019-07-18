#!/usr/bin/python3
# Leon Graumans
#
# script for counting/scraping 4chan boards

from __future__ import print_function
from nltk import word_tokenize
from bs4 import BeautifulSoup

import os
import re
import pickle
import requests
import basc_py4chan


def get_archived_ids(board):
    archived_ids = []
    if board == 'b':
        return archived_ids

    href = 'http://boards.4chan.org/' + board + '/archive'
    html = requests.get(href).content
    soup = BeautifulSoup(html, 'html.parser')
    result = soup.find('table', { 'class' : 'flashListing'})

    for row in result.findAll('tr'):
        cells = row.findAll('td')
        if len(cells) == 3 and cells[0].find(text=True) != 'No.': # skip header
            archived_ids.append(int(cells[0].find(text=True)))

    return archived_ids

def remove(text):
    noise = [r'>>\S+', r'\S+>>']
    for item in noise:
        text = re.sub(item, '', text)
    return text

def main():
    wordcount = 0
    file = open('4chan/txt_4chan.txt', 'a')

    try:
        with open('4chan/json_4chan.pickle', 'rb') as f:
            comment_dict = pickle.load(f)
            print('Using existing dictionary')
    except Exception:
        comment_dict = {}
        print('Using new dictionary')

    # list_of_boards = ['b', 'r9k', 's4s', 'pol']
    # list_of_boards1 = ['lgbt', 'x', 'adv', 'news', 'vip', 'qa']
    pol = ['pol']
    for b in pol:
        board = basc_py4chan.Board(b)

        thread_ids = board.get_all_thread_ids()
        archived_ids = get_archived_ids(b)
        all_ids = thread_ids + archived_ids
        for thread_id in all_ids:
            thread = board.get_thread(thread_id)
            try:
                for text in thread.all_posts:
                    id = text.post_id
                    if id in comment_dict:
                        continue

                    text = remove(text.text_comment)
                    text = os.linesep.join([s for s in text.splitlines() if s]) # remove empty lines
                    tokens = word_tokenize(text)
                    wordcount += len(tokens)

                    comment_dict[id] = {'body': text, 'thread_id': thread_id}
                    file.write(text)

            except Exception:
                print('Failed to load comments\n')
                continue

            print(b, thread_id)
            print('Total: {:,}\n'.format(wordcount))

    file.close()

    with open('4chan/json_4chan.pickle', 'wb') as f:
        pickle.dump(comment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
