

import gc
import re

from sklearn import feature_extraction
import nltk

class TextCleanser:
    def __init__(self, texts) -> None:
        gc.collect()
        self.texts = ( sentence.lower() for sentence in texts)

    def remove_html_tags(self):
        gc.collect()
        xclean = []
        pattern = re.compile('<.*?>')
        for sentence in self.texts:
            temptext = pattern.sub(r'', sentence)
            xclean.append(temptext)

        self.texts = iter(xclean)
        return xclean

    def remove_urls(self):
        gc.collect()
        xclean = []
        pattern = re.compile(r'https?://\S+|www\.\S+')
        for sentence in self.texts:
            temptext = pattern.sub(r'', sentence)
            xclean.append(temptext)

        self.texts = iter(xclean)
        return xclean

    def remove_punctuations(self):
        gc.collect()
        xclean = []
        excluders = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        for sentence in self.texts:
            temptext = sentence.translate( str.maketrans('', '', excluders) )
            xclean.append(temptext)

        self.texts = iter(xclean)
        return xclean

    def remove_stop_words(self, custom_words=[]):
        gc.collect()
        self.stop_words = feature_extraction.text.ENGLISH_STOP_WORDS.union(custom_words)
        xclean = []

        for text in self.texts:
            words = list()
            for sentence in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sentence):
                    if not word in self.stop_words and word.isalpha():
                        words.append(word)
            sentence = " ".join(words)
            xclean.append(sentence)

        self.texts = iter(xclean)
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
        for text in self.texts:
            temptext = emoji_pattern.sub(r"", text)
            xclean.append(temptext)
        
        self.texts = iter(xclean)
        return xclean

    def perform_stem(self):
        gc.collect()
        xclean = []
        ps = nltk.stem.PorterStemmer()
        for text in self.texts:
            temptext = " ".join([ps.stem(word) for word in nltk.word_tokenize(text)])
            xclean.append(temptext)
        
        self.texts = iter(xclean)
        return xclean

    def perform_lemmatize(self):
        gc.collect()
        xclean = []
        wnl = nltk.stem.WordNetLemmatizer()
        for text in self.texts:
            temptext = " ".join([wnl.lemmatize(word)
                                for word in nltk.word_tokenize(text)])
            xclean.append(temptext)

        self.texts = iter(xclean)
        return xclean



    
