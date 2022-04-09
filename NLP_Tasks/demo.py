"""
# _* coding: utf8 *_

filename: cleanser.py

@author: sounishnath
createdAt: 2022-04-09 22:06:49
"""


import gc
import re

import nltk
from nltk.corpus import wordnet
from sklearn import feature_extraction
from tqdm import tqdm


class TextCleanser:
    def __init__(self, texts) -> None:
        gc.collect()
        self.texts = [sentence.lower() for sentence in texts]

    def remove_html_tags(self):
        gc.collect()
        xclean = []
        pattern = re.compile('<.*?>')
        for sentence in tqdm(self.texts):
            temptext = pattern.sub(r'', sentence)
            xclean.append(temptext)

        self.texts = xclean
        return xclean

    def remove_urls(self):
        gc.collect()
        xclean = []
        pattern = re.compile(r'https?://\S+|www\.\S+')
        for sentence in tqdm(self.texts):
            temptext = pattern.sub(r'', sentence)
            xclean.append(temptext)

        self.texts = xclean
        return xclean

    def remove_punctuations(self):
        gc.collect()
        xclean = []
        excluders = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for sentence in tqdm(self.texts):
            temptext = sentence.translate(str.maketrans('', '', excluders))
            xclean.append(temptext)

        self.texts = xclean
        return xclean

    def remove_stop_words(self, custom_words=[]):
        gc.collect()
        self.stop_words = feature_extraction.text.ENGLISH_STOP_WORDS.union(
            custom_words)
        xclean = []

        for text in tqdm(self.texts):
            words = list()
            for sentence in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sentence):
                    if not word in self.stop_words and word.isalpha():
                        words.append(word)
            sentence = " ".join(words)
            xclean.append(sentence)

        self.texts = xclean
        return xclean

    def remove_emoji_or_non_unicodes(self):
        gc.collect()
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        xclean = []
        for text in tqdm(self.texts):
            temptext = emoji_pattern.sub(r"", text)
            xclean.append(temptext)

        self.texts = xclean
        return xclean

    def perform_stem(self):
        gc.collect()
        xclean = []
        ps = nltk.stem.PorterStemmer()
        for text in tqdm(self.texts):
            temptext = " ".join([ps.stem(word)
                                for word in nltk.word_tokenize(text)])
            xclean.append(temptext)

        self.texts = xclean
        return xclean

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def perform_lemmatize(self):
        gc.collect()
        xclean = []
        wnl = nltk.stem.WordNetLemmatizer()
        for text in tqdm(self.texts):
            temptext = " ".join([wnl.lemmatize(word, self.get_wordnet_pos(word))
                                for word in nltk.word_tokenize(text)])
            xclean.append(temptext)

        self.texts = xclean
        return xclean


    def clean_everything_by_process(self, c_type='lemma'):
        self.remove_urls()
        print('removing urls done...')
        self.remove_html_tags()
        print('removing html tags done...')
        self.remove_punctuations()
        print('removing punctuations done...')
        self.remove_emoji_or_non_unicodes()
        print('removing non unicoded characters done...')

        if c_type == 'lemma':
            self.perform_lemmatize()
        elif c_type == 'stem':
            self.perform_stem()
        print(f'transforming texts to {c_type} done...')

        xclean = self.remove_stop_words()
        print('removing stop-words done...')

        self.texts = xclean
        return xclean
