# imports needed and logging
import gensim 
import logging
import sys
# read in some helpful libraries
import re                         # regular expression
import numpy as np
from gensim.models import KeyedVectors
from gensim.parsing.porter import PorterStemmer
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Text8Corpus
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

class w2v():
    def __init__(self,vector_size=250, min_count=1, epochs=100):
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.bigram = None

    def most_similar(self,word,topn=5):
        if self.stem:
            p = PorterStemmer()
            word = p.stem(word)
        list_of_words = self.model.most_similar(word,topn=topn)
        return [word[0] for word in list_of_words]

    def __getitem__(self, item):
        return self.model[item]

    def cosine_sim(self,w1,w2):
        if self.stem:
            p = PorterStemmer()
            w1 = p.stem(w1)
            w2 = p.stem(w2)
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
                yield list_of_words

    def prepare_bigram(self, corpus_filename,bigram_filename, common_terms=frozenset([]), min_count=1,threshold=5):
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
        # build vocabulary and train model
        model = gensim.models.Word2Vec(
            documents,
            size=self.vector_size,
            window=14,
            min_count=self.min_count,
            workers=10)
        model.train(documents, total_examples=len(documents), epochs=self.epochs)
        if(outputfile != ''):
            path_to_model = get_tmpfile(outputfile)
            model.wv.save(path_to_model)
        self.model = model
        self.vocab = model.wv.vocab
        # if ('diabet' in self.vocab):
        #     logging.error("after train, its here man")
        # else:
        #     logging.error("after train vocab deo co")

        return self.vocab

    def doc_2_vec(self,doc):
        doc = re.sub("-", "_", str(doc))
        list_of_words = gensim.utils.simple_preprocess(doc)
        if (self.bigram is not None):
            # Further detect phrases
            list_of_words = self.bigram[list_of_words]
        if self.stem:
            p = PorterStemmer()
            p.stem_documents(list_of_words)
        words_vec = np.array([self[word] for word in list_of_words if word in self.vocab])
        if(len(words_vec) == 0):
            #logging.warning('cant find ',doc)
            #logging.warning('cant find',list_of_words)
            return np.zeros(self.model.vector_size)
        return words_vec.mean(axis=0)

    @classmethod
    def load(cls,filename,bigram_filename='',stem=True):
        path_to_model = get_tmpfile(filename)
        obj = w2v()
        obj.model = KeyedVectors.load(path_to_model, mmap='r')
        obj.vocab = obj.model.vocab
        if(bigram_filename != ''):
            path_to_bigram = get_tmpfile(bigram_filename)
            obj.bigram = Phraser.load(path_to_bigram)
        else:
            obj.bigram = None
        obj.stem = stem
        return obj

