"""
# _* coding: utf8 *_

filename: tokenizer.py

@author: sounishnath
createdAt: 2022-12-13 13:36:50
"""

import re
import string
import typing

import nltk
import numpy as np


class Tokenizer:
    def __init__(self, texts: typing.List[str], pad_length: int = 20) -> None:
        self.__rawtexts = texts.copy()
        self.pad_length = pad_length
        self.__lemmer = nltk.stem.WordNetLemmatizer()
        self.texts = [" ".join(self.___tokenize(sent)) for sent in self.__rawtexts]
        self.vocabset = self.__build_vocab(self.texts)
        self.word_to_idx = dict(
            (word, index) for index, word in enumerate(self.vocabset)
        )
        self.idx_to_word = dict(
            (index, word) for index, word in enumerate(self.vocabset)
        )

    def __build_vocab(self, texts: typing.List[str]):
        vocab_set = set()

        for text in texts:
            for sent in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sent):
                    vocab_set.add(word)

        vocab_set.add("[PAD]")
        vocab_set.add("[SEP]")
        vocab_set.add("[CLS]")
        vocab_set.add("[UNK]")

        return vocab_set

    def ___tokenize(self, sentence: str):
        words = nltk.word_tokenize(sentence.lower().strip())[: self.pad_length]
        wor = ["[SEP]"]
        for word in words:
            if not word in set(string.punctuation) or not word in ["[UNK]", "[PAD]"]:
                wor.append(self.__lemmer.lemmatize(word))
        wor.append("[SEP]")
        [wor.append("[PAD]") for _ in range(self.pad_length - len(wor))]
        return wor

    def normalizeString(self, s: str):
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def tokenize(self, text: str):
        text = self.normalizeString(text.lower().strip())
        words = [
            self.__lemmer.lemmatize(word)
            for word in nltk.word_tokenize(text.lower().strip())
            if not word in set(string.punctuation)
        ][: self.pad_length]

        [words.append("[PAD]") for _ in range(self.pad_length - len(words))]
        ids = [self.word_to_idx.get(word, self.word_to_idx["[UNK]"]) for word in words]

        return {
            "text": text,
            "tokenized_text": np.array(words),
            "token_type_ids": np.array(ids),
        }

    def one_hot_encode(self, text: str):
        text = self.normalizeString(text.lower().strip())
        words = [
            self.__lemmer.lemmatize(word)
            for word in nltk.word_tokenize(text.lower().strip())
            if not word in ["[UNK]", "[PAD]"] and not word in set(string.punctuation)
        ][: self.pad_length]

        [words.append("[PAD]") for _ in range(self.pad_length - len(words))]
        ids = [self.word_to_idx.get(word, self.word_to_idx["[UNK]"]) for word in words]

        return {
            "text": text,
            "tokenized_text": np.array(words),
            "token_type_ids": np.array(ids),
        }
