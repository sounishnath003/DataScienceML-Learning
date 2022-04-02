
import collections
import gc

import nltk
import numpy as np


class TextTokenizer:
    def __init__(self, data, num_words=None, sequence_length=None) -> None:
        self.data = np.array(data)
        self.num_words = num_words
        self.sequence_length = sequence_length

    def build_vocabulary(self):
        gc.collect()
        freqDict = collections.Counter()
        for sentence in self.data:
            for word in nltk.word_tokenize(sentence):
                freqDict[word] += 1

        common_words = freqDict.most_common(self.num_words)
        self.vocabulary = dict()

        for index, (word, freq) in enumerate(common_words):
            self.vocabulary[word] = index+1

        self.vocabulary['<UNK>'] = len(self.vocabulary) + 1
        return self.vocabulary

    def vectorize_sentences_by_vocabulary(self):
        gc.collect()
        xtokenized = []
        for sentence in self.data:
            tempsentence = self.vectorize_single_sentence(sentence)
            xtokenized.append(tempsentence)

        self.data_tokenized = xtokenized
        return xtokenized

    def vectorize_single_sentence(self, sentence):
        gc.collect()
        tempsentence = []
        for word in nltk.word_tokenize(sentence):
            if not word in self.vocabulary:
                tempsentence.append(self.vocabulary.get('<UNK>'))
            else:
                tempsentence.append(self.vocabulary.get(word))
        return tempsentence

    def roundtrip_check(self, sentence):
        temp = []
        for word in nltk.word_tokenize(sentence):
            if word in self.vocabulary:
                temp.append(word)
            else:
                temp.append('<UNK>')
        return temp

    def pad_sequences(self, pad_type="post"):
        gc.collect()
        d_padded = []

        for tokenized in self.data_tokenized:
            if pad_type.lower() == "post":
                while len(tokenized) < self.sequence_length:
                    tokenized.insert(len(tokenized), 0)
                d_padded.append(tokenized)
            else:
                while len(tokenized) < self.sequence_length:
                    tokenized.insert(0, 0)
                d_padded.append(tokenized)

        self.data_padded = d_padded
        return d_padded
