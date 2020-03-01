import pandas as pd
import numpy as np
import re
import os
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from time import gmtime, strftime
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(txt, regex, replacement):
	'''
	cleans each string in the list to remove unnecessary characters

	Parameters
	----------
	txt: str
		text data to be cleaned
	regex: str
		regular expression indicating what pattern to replace
	replacement: string
		substitute for the pattern in the regex

	Returns
	-------
	cleaned str

	'''
	return re.sub(regex, replacement, str(txt))

def set_stopwords(li_extend):
	'''
	initiates the stopwords in english and adds additional words if desired

	Parameters
	----------
	li_extend: list of strs
		contains words to add to the stop words list

	Returns
	-------
	a list of stop words in string format

	'''
	li_stop_words = stopwords.words('english')
	if li_extend is not None:
		li_stop_words.extend(li_extend)
	return li_stop_words

def tokenizer(txt, li_stop_words):
	'''
	applies nltk word_tokenize function to str and removes stop words

	Parameters
	----------
	txt: str
		text data to be tokenized
	li_stop_words: list
		list of words to be removed from the tokens

	Returns
	-------
	cleaned list of words in string format

	'''
	tokens = word_tokenize(txt.lower())
	return [x for x in tokens if x not in li_stop_words]

def lemmatizer(corpus):
	'''
	applies nltk WordNetLemmatizer to list of tokens

	Parameters
	----------
	corpus: list
		list of tokens to be lemmatized

	Returns
	-------
	list of lemmatized words in string format

	'''
	lmtzr=WordNetLemmatizer()
	return[lmtzr.lemmatize(word, get_wordnet_pos(word)) for word in corpus]

def get_wordnet_pos(word):
	'''
	Map POS tag to first character WordNetLemmatizer lemmatize() accepts

	Parameters
	----------
	word: str
		word to be lemmatized

	Returns
	-------
	str with POS tag
	'''
	tag = pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ,
			"N": wordnet.NOUN,
			"V": wordnet.VERB,
			"R": wordnet.ADV}
	return tag_dict.get(tag, wordnet.NOUN)

def stemmer(corpus):
	'''
	applies nltk SnowballStemmer to list of tokens

	Parameters
	----------
	corpus: list
		list of tokens to be stemmed

	Returns
	-------
	list of stemmed words in string format

	'''
	stmr=SnowballStemmer('english')
	return[stmr.stem(word) for word in corpus]

def text_cleaner(txt, di_regex, li_stop_words, fl_lemmatize, fl_stemmer):
	'''
	cleans, tokenizes, and optionally lemmatizes a string of text

	Parameters
	----------
	txt: str
		text to process
	di_regex: dictionary (key = regex: value = replacement)
		regular expressions used to clean the data
	li_stop_words: list
		list of words to be removed from text
	fl_lemmatize: bool
		indicates whether to run the lemmatizer function

	Returns
	-------
	txt: list
		cleaned tokens

	'''
	for pattern, value in di_regex.items():
		txt=clean_text(txt, pattern, value)
	txt=tokenizer(txt, li_stop_words)
	if fl_lemmatize:
		txt=lemmatizer(txt)
	if fl_stemmer:
		txt=stemmer(txt)
	return ' '.join(txt)

def text_preprocessing(corpus, di_regex, li_stop_words, fl_lemmatize, fl_stemmer):
	'''
	cleans, tokenizes, and optionally lemmatizes a string of text

	Parameters
	----------
	txt: str
		text to process
	di_regex: dictionary (key = regex: value = replacement)
		regular expressions used to clean the data
	li_stop_words: list
		list of words to be removed from text
	fl_lemmatize: bool
		indicates whether to run the lemmatizer function

	Returns
	-------
	txt: list
		cleaned tokens

	'''
	corpus=[text_cleaner(txt, di_regex, li_stop_words, fl_lemmatize, fl_stemmer) for txt in corpus]
	return corpus
