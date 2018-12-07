# imports needed and logging
import gensim 
import logging
import sys
# read in some helpful libraries
import re                         # regular expression
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Text8Corpus
from gensim.test.utils import  get_tmpfile
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

class w2v():
    def __init__(self,vector_size=250, min_count=1, epochs=100):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.bigram = None

    def most_similar(self,word,topn=5):
        list_of_words = self.model.most_similar(word,topn=topn)
        return [word[0] for word in list_of_words]

    def __getitem__(self, item):
        return self.model[item]

    def cosine_sim(self,w1,w2):
        vectA = np.array(self.model[w1])
        vectB = np.array(self.model[w2])
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

    def prepare_bigram(self, corpus_file_name, common_terms=frozenset([]), min_count=1,threshold=5):
        # Read file and return sentences as a list of lists of string
        # where each list of string represents a sentence
        sentences = Text8Corpus((corpus_file_name))

        # Detect and group phrases
        phrases = Phrases(sentences, min_count=min_count, threshold=threshold, common_terms=common_terms)
        bigram = Phraser(phrases)
        self.bigram = bigram

    def fit(self,corpus_file_name,outputfile=''):
        documents = list(self.read_corpus(corpus_file_name))
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            documents,
            size=self.vector_size,
            window=10,
            min_count=self.min_count,
            workers=10)
        model.train(documents, total_examples=len(documents), epochs=self.epochs)
        if(outputfile != ''):
            path_to_model = get_tmpfile(outputfile)
            model.wv.save(path_to_model)
        self.model = model
        self.vocab = model.wv.vocab

    def doc_2_vec(self,doc):
        print('cant find ', doc)
        doc = re.sub("-", "_", str(doc))
        list_of_words = gensim.utils.simple_preprocess(doc)
        if (self.bigram is not None):
            # Further detect phrases
            list_of_words = self.bigram[list_of_words]
        words_vec = np.array([self[word] for word in list_of_words if word in self.vocab])
        if(len(words_vec) == 0):
            logging.warning('cant find ',doc)
            #logging.warning('cant find',list_of_words)
            return np.zeros(self.model.vector_size)
        return words_vec.mean(axis=0)

    @classmethod
    def load(cls,filename):
        path_to_model = get_tmpfile(filename)
        cls.model = KeyedVectors.load(path_to_model, mmap='r')
        cls.vocab = cls.model.vocab
        return cls()

