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


subreddit_list1 = [ 'atheism',
                    'Conservative',
                    'conspiracy',
                    'GenderCritical',
                    'The_Donald',
                    'KotakuInAction',
                    'mensrights',
                    'AlternativeHealth',
                    'ClimateSkeptics',
                    'Overpopulation',
                    'Collapse',
                    'Communism',
                    'ShitLiberalsSay',
                    'LateStageCapitalism',
                    'Socialism',
                    'Bad_Cop_No_Donut',
                    'Anarchism',
                    'Anarcho_Capitalism',
                    'DarkEnlightenment',
                    'DebateNazism',
                    'AznIdentity',
                    'Bakchodi',
                    'EasternSunRising',
                    'Prolife',
                    'TrueChristian',
                    'UnpopularOpinion',
                    'ShitRedditSays',
                    'creepyPMs',
                    'Technology',
                    'ForeverAlone',
                    'theredpill',
                    'Holocaust',
                    '911Truth',
                    'FULLCOMMUNISM',
                    'DebateFascism',
                    'HateSubsInAction',
                    'DebateAltRight',
                    'TumblrInAction',
                    'PussyPassDenied',
                    'PussyPass',
                    'CringeAnarchy',
                    'Drama',
                    'watchpeopledie',
                    'MGTOW',
                    'aznidentity',
                    'ShitPoliticsSays',
                    'metacanada',
                    'Imgoingtohellforthis',
                    'dankmemes',

]

subreddit_list2 = [
                    'The_Congress',
                    'GreatAwakening',
                    'LeftWithSharpEdge',
                    'AntiPOZI',
                    'altright',
                    'European',
                    'Physical_Removal',
                    'PublicHealthWatch',
                    'WhiteRights',
                    'AganistGayMarriage',
                    'Incel',
                    'PhilosophyOfRape',
                    'fatpeoplehate',
]

subreddit_list3 = [
                    'ShitRConservativeSays',
                    'ConspiracyII',
                    'TopMindsOfReddit',
                    'TrollXChromosomes',
                    'Ice_Poseidon',
                    'Ice_Poseidon2',
                    'whitebeauty',
]


def url_replacer(text):
    url = [r'http\S+', r'\S+https', r'\S+http']
    for item in url:
        text = re.sub(item, '<URL>', text)
    return text


def main():
    wordcount = 0
    comment_list = []
    file = open('txt_reddit_hot.txt', 'w')

    try:
        with open('json_reddit_con.pickle', 'rb') as f:
            comment_dict = pickle.load(f)
            print('Using existing dictionary')
    except Exception:
        comment_dict = {}
        print('Using new dictionary')

    for topic in subreddit_list1:

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


        for submission in subreddit.hot(limit=None):
            submission_id = submission.id
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                id = comment.id

                if id in comment_dict:
                    continue

                parent_id = comment.parent_id
                subreddit_id = comment.subreddit_id

                text = comment.body
                text = os.linesep.join([s for s in text.splitlines() if s]) # remove empty lines
                text = url_replacer(text)

                tokens = word_tokenize(text)
                wordcount += len(tokens)

                # comment_list.append(text)
                comment_dict[id] = {'body': text, 'parent_id': parent_id, 'subreddit_id': subreddit_id}
                file.write(text)


        print('Total: {:,}\n'.format(wordcount))

    file.close()

    with open('json_reddit_hot.pickle', 'wb') as f:
        pickle.dump(comment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
