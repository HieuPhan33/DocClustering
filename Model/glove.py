#!/usr/bin/env python
# coding: utf-8

# In[69]:


import itertools
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser
from glove import Corpus, Glove
import gensim
import logging
from pathlib import Path
from gensim.test.utils import get_tmpfile
import pandas as pd
import numpy as np
import re


# In[83]:
class glove():
    def __init__(self,vector_size=250, min_count=1, epochs=100,learning_rate=0.02):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.bigram = None

    def most_similar(self,word,topn=5):
        list_of_words = self.model._similarity_query(self[word],topn+1)
        return [list_of_words[i][0] for i in range(1,topn+1)]

    def __getitem__(self, item):
        return self.model.word_vectors[self.model.dictionary[item]]

    def cosine_sim(self,w1,w2):
        vectA = np.array(self[w1])
        vectB = np.array(self[w2])
        return (vectA.dot(vectB))/(np.linalg.norm(vectA) * np.linalg.norm(vectB))

    def read_corpus(self, corpus_file_name):
        with open(corpus_file_name, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = re.sub("-", "_", str(line))
                list_of_words = gensim.utils.simple_preprocess(line)
                if (self.bigram is not None):
                    # Further detect phrases
                    list_of_words = self.bigram[list_of_words]
                yield list_of_words

    def prepare_bigram(self, corpus_filename,bigram_filename, common_terms=frozenset([]), min_count=1,threshold=5):
        # Read file and return sentences as a list of lists of string
        # where each list of string represents a sentence
        sentences = Text8Corpus(corpus_filename)

        # Detect and group phrases
        phrases = Phrases(sentences, min_count=min_count, threshold=threshold, common_terms=common_terms)
        bigram = Phraser(phrases)
        path_to_bigram = get_tmpfile(bigram_filename)
        bigram.save(path_to_bigram)
        self.bigram = bigram

    def fit(self,corpus_file_name,outputfile=''):
        # Use bigram to detect phrases and return well-structured list of sentences with phrases
        sentences = list(self.read_corpus(corpus_file_name))
        corpus = Corpus()
        corpus.fit(sentences, window=10)
        glove = Glove(no_components=self.vector_size, learning_rate=self.learning_rate)
        glove.fit(corpus.matrix, epochs=self.epochs, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        if(outputfile != ''):
            path_to_model = get_tmpfile('glove.model')
            glove.save(path_to_model)
        self.model = glove
        self.vocab = list(glove.dictionary)

    def doc_2_vec(self,doc):
        doc = re.sub("-", "_", str(doc))
        list_of_words = gensim.utils.simple_preprocess(doc)
        if (self.bigram is not None):
            # Further detect phrases
            list_of_words = self.bigram[list_of_words]
        words_vec = np.array([self[word] for word in list_of_words if word in self.vocab])
        if(len(words_vec) == 0):
            return np.zeros(self.model.no_components)
        return words_vec.mean(axis=0)

    @classmethod
    def load(cls,filename,bigram_filename=''):
        obj = glove()
        path_to_model = get_tmpfile(filename)
        obj.model = Glove.load(path_to_model)
        obj.vocab = obj.model.dictionary
        if(bigram_filename != ''):
            path_to_bigram = get_tmpfile(bigram_filename)
            obj.bigram = Phraser.load(path_to_bigram)
        return obj

