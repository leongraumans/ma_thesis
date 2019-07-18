#!/usr/bin/python3
# Leon Graumans
#
# script for counting/scraping 8chan boards

from __future__ import print_function
from nltk import word_tokenize
from bs4 import BeautifulSoup

import os
import re
import pickle
import requests
import py8chan

def get_archived_ids(board):
	archived_ids = []
	if board == 'b':
		return archived_ids

	href = 'https://8ch.net/' + board + '/archive'
	html = requests.get(href).content
	soup = BeautifulSoup(html, 'html.parser')
	result = soup.find('table', { 'class' : 'flashListing'})
	links = [a.get('href') for a in soup.find_all('a', href=True)]
	print(links)

	for link in soup.findAll('a', attrs={'href': re.compile("^pol/res/")}):
		print(link.get('href'))
	exit()


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
	file = open('8chan/txt_8chan.txt', 'a')

	try:
		with open('8chan/json_8chan.pickle', 'rb') as f:
			comment_dict = pickle.load(f)
			print('Using existing dictionary')
	except Exception:
		comment_dict = {}
		print('Using new dictionary')


	pol = ['pol', 'leftypol', 'b']

	for b in pol:
		board = py8chan.Board(b)

		thread_ids = board.get_all_thread_ids()

		scraped_ids = []
		with open('8ch_ids.txt', 'r') as f:
			for line in f:
				scraped_ids.append(int(line))

		all_ids = thread_ids + scraped_ids

		print(len(thread_ids))
		print(len(scraped_ids))

		for thread_id in thread_ids:
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

	with open('8chan/json_8chan.pickle', 'wb') as f:
		pickle.dump(comment_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
	main()
