#!/usr/bin/python3
# Leon Graumans
#
# script for counting/scraping all comments of specific subreddit_list
# add subreddits to subreddit_list
#
# needs Reddit API
# https://praw.readthedocs.io/en/latest/

import os
import re
import pickle
import pprint
import praw
import prawcore
from praw.models import MoreComments
from nltk import word_tokenize

reddit = praw.Reddit(client_id='CLIENT_ID',
                     client_secret='CLIENT_SECRET',
                     user_agent='USER_AGENT',
                     username='USERNAME',
                     password='PASSWORD')

reddit.read_only = True


subreddit_list_np = [
                    'funny',
                    'AskReddit',
                    'todayilearned',
                    'worldnews',
                    'Science',
                    'pics',
                    'gaming',
                    'IAmA',
                    'videos',
                    'movies',
                    'aww',
                    'Music',
                    'blog',
                    'gifs',
                    'news',
                    'explainlikeimfive',
                    'askscience',
                    'EarthPorn',
                    'books',
                    'television',
                    'mildlyinteresting',
                    'Showerthoughts',
                    'LifeProTips',
                    'space',
                    'DIY',
                    'Jokes',
                    'gadgets',
                    'nottheonion',
                    'sports',
                    'food',
                    'tifu'
]


def url_replacer(text):
    url = [r"http\S+", r"\S+https", r"\S+http"]
    for item in url:
        text = re.sub(item, '<URL>', text)
    return text


def main():
    wordcount = 0
    comment_list = []
    comment_dict = {}
    file = open('txt_reddit_np2.txt', 'w')
    feedback = open('txt_reddit_np2_feedback.txt', 'w')
    ids = open('txt_reddit_np2_ids.txt', 'w')
    for topic in subreddit_list1:

        feedback.write(topic)

        try:
            subreddit = reddit.subreddit(topic)
            print(subreddit.title)
        except prawcore.UnavailableForLegalReasons:
            print('Unavailable: ', topic, '\n')
            continue
        except prawcore.exceptions.NotFound:
            print('Banned: ', topic, '\n')
            continue
        except prawcore.exceptions.Forbidden:
            print('Quarantained')
            try:
                subreddit.quaran.opt_in()
                print(subreddit.title, '\n')
            except Exception:
                print('Failed to load: ', topic, '\n')
                continue

        continue

        for submission in subreddit.top('year'):
            submission_id = submission.id
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                id = comment.id
                parent_id = comment.parent_id
                subreddit_id = comment.subreddit_id

                text = comment.body
                text = os.linesep.join([s for s in text.splitlines() if s]) # remove empty lines
                text = url_replacer(text)

                tokens = word_tokenize(text)
                wordcount += len(tokens)

                comment_dict[id] = {'body': text, 'parent_id': parent_id, 'subreddit_id': subreddit_id}
                file.write(text)
                id_newline = id + '\n'
                ids.write(id_newline)

        wc_newline = str(wordcount) + '\n'
        feedback.write(wc_newline)

        print('Total: {:,}\n'.format(wordcount))

    file.close()
    ids.close()
    feedback.close()

    with open('json_reddit_np2.pickle', 'wb') as f:
        pickle.dump(comment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
