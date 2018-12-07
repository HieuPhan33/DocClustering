#!/usr/bin/env python
# coding: utf-8

# In[22]:


from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.porter import PorterStemmer
from gensim.models.word2vec import Text8Corpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import gensim
import logging
import io
import numpy as np
from pathlib import Path
# read in some helpful libraries
import pandas as pd               # pandas dataframe
import re                         # regular expression
from gensim.test.utils import get_tmpfile
import re
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# In[12]:
class d2v():
    def __init__(self,vector_size=250, min_count=1, epochs=100,learning_rate = 0.02):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.bigram = None

    def most_similar(self,word,topn=5):
        if self.stem:
            p = PorterStemmer()
            word = p.stem(word)
        list_of_words = self.model.most_similar(word,topn=topn)
        return [word[0] for word in list_of_words]

    def __getitem__(self, item):
        return self.model.wv[item]

    def cosine_sim(self,w1,w2):
        vectA = np.array(self.model[w1])
        vectB = np.array(self.model[w2])
        return (vectA.dot(vectB))/(np.linalg.norm(vectA) * np.linalg.norm(vectB))

    def read_corpus(self, corpus_file_name,stem):
        with open(corpus_file_name, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = re.sub("-", "_", str(line))
                list_of_words = gensim.utils.simple_preprocess(line)
                if (self.bigram is not None):
                    # Further detect phrases
                    list_of_words = self.bigram[list_of_words]
                if stem:
                    p = PorterStemmer()
                    list_of_words = p.stem_documents(list_of_words)
                yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    def prepare_bigram(self, corpus_filename,bigram_filename, common_terms=frozenset([]), min_count=1,threshold=1.5):
        # Read file and return sentences as a list of lists of string
        # where each list of string represents a sentence
        sentences = Text8Corpus((corpus_filename))

        # Detect and group phrases
        phrases = Phrases(sentences, min_count=min_count, threshold=threshold, common_terms=common_terms)
        bigram = Phraser(phrases)
        path_to_bigram = get_tmpfile(bigram_filename)
        bigram.save(path_to_bigram)
        self.bigram = bigram

    def fit(self,corpus_file_name,outputfile='',stem=True):
        self.stem = stem
        documents = list(self.read_corpus(corpus_file_name,stem))
        model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size, min_count=self.min_count,
                                              epochs=self.epochs,alpha=self.learning_rate)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        if(outputfile != ''):
            path_to_model = get_tmpfile(outputfile)
            model.save(path_to_model)
        self.model = model
        self.vocab = model.wv.vocab

    def doc_2_vec(self,doc):
        doc = re.sub("-", "_", str(doc))
        list_of_words = gensim.utils.simple_preprocess(doc)
        if (self.bigram is not None):
            # Further detect phrases
            list_of_words = self.bigram[list_of_words]
        if self.stem:
            p = PorterStemmer()
            p.stem_documents(list_of_words)
        valid_words = [word for word in list_of_words if word in self.vocab]
        if(len(valid_words) == 0):
            #logging.warning('cant find ',doc)
            #logging.warning('cant find',list_of_words)
            return np.zeros(self.model.vector_size)
        return self.model.infer_vector(list_of_words,alpha=0.02,steps=100)

    @classmethod
    def load(cls,filename,bigram_filename = '',stem=True):
        path_to_model = get_tmpfile(filename)
        obj = d2v()
        obj.model = Doc2Vec.load(path_to_model)
        obj.vocab = obj.model.wv.vocab
        if(bigram_filename != ''):
            path_to_bigram = get_tmpfile(bigram_filename)
            obj.bigram = Phraser.load(path_to_bigram)
        obj.stem = stem
        return obj
