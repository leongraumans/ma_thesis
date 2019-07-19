# Master thesis L.R.N. (Leon) Graumans
## Abusive Language Detection in Online User Content using Polarized Word Embeddings

### Structure
- **classifier** Python scripts used for the classification part of this thesis:
	- **biLSTM.py**: contains the bidirectional LSTM model.
	- **classifier.py**: used as the main program, contains the SVM model which uses ngrams or embeddings as its feature. Also calls the BiLSTM, if specified in arguments.
	- **features.py**: contains features used in classifier.py.
	- **helperFunctions.py**: contains small helpful functions for classifier.py, such as read functions for the data sets and embeddings.
	- **jsd.py**: file for calculating jensen shennon divergence between two datasets.
	- **transformers.py**: contains transformers used in classifier.py.
- **datasets** Data sets used for this thesis:
	- **HatEval**: dataset used for the SemEval 2019: Task 5 - A. Detection of hate speech.
	- **OffensEval**: dataset used for the SemEval 2019: Task 6 - A. Detection of offensive language.
	- **WaseemHovy**: dataset by Waseem & Hovy. Detection of racism and sexism.
- **scraper** Scripts used for scraping data to generate word embeddings:
	- **knns.py**: script for printing the k nearest neighbours for 20 hand-picked keywords in word embeddings.
	- **list_of_hashtags.txt**: list of 310 hashtags used for scraping controversial tweets.
	- **preprocess.py**: script for preprocessing data, using regular expressions.
	- **ruby-preprocess.rb**: script for preprocessing data, using regular expressions, used for GloVe embeddings.
	- **scrape_4chan_general.py**: script for scraping data from 4chan.
	- **scrape_4chan_polarised.py**: script for scraping data from 4chan.
	- **scrape_8chan_polarised.py**: script for scraping data from 8chan.
	- **scrape_reddit_general.py**: script for scraping data from reddit.
	- **scrape_reddit_polarised.py**: script for scraping data from reddit.
	- **scrape_twitter.py**: script for scraping data from twitter.

### Data
- [Waseem & Hovy 2016](https://github.com/ZeerakW/hatespeech)
- [OffensEval Task A](https://competitions.codalab.org/competitions/20011)
- [HatEval English Task A](https://competitions.codalab.org/competitions/19935)

### Abstract
Anytime one engages online, there is always a serious risk that he or she may be the target of toxic and abusive speech. To combat such behaviour, many internet companies have terms of services on these platforms that typically forbid hateful and harassing speech. However, the increasing volume of online data requires that ways are found to classify online content automatically. 

In this work, we present a way to automatically detect abusive language in online user content. We create polarized word embeddings from controversial social media data to better model phenomena like offensive language, hate speech and other forms of abusive language. We compare these polarized embedding representations towards more standard generic embeddings, which are in principle a representative of the English language.

Two machine learning models are used to measure the contribution of our polarized word embeddings to the detection of abusive language. We found that the polarized Reddit embeddings, created with the FastText algorithm, proved superior to the other source-driven representations. When applied in a bidirectional LSTM model, our polarized embeddings outperformed the pre-trained generic embeddings, even when used across multiple data sets.

### Acknowledgements
*Most of these scripts for classification are re-used and modified, original versions from:*

- [Nissim et al, GermEval](https://github.com/malvinanissim/germeval-rug)
- [Balint Hompot, OffensEval](https://github.com/BalintHompot/RUG_Offenseval)
- [Grunn2019 at SemEval 2019 Task 5](https://bitbucket.org/grunn2018/sharedhate_repo)
