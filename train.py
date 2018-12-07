from Model.d2v import d2v
from Model.glove import glove
from Model.w2v import w2v
import logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--size', default=350,
                    help='Vector size of your output vector')
parser.add_argument('--epochs', default=400,
                    help='Number of epochs')
parser.add_argument('--corpus',default='corpus/merged.txt',
                    help='Training dataset')
args = parser.parse_args()
vector_size = args.size
epochs = args.epochs
corpus_file_name = args.corpus

def main():
    w2v_model = w2v(vector_size=vector_size, min_count=8, epochs=epochs)
    w2v_model.prepare_bigram(corpus_file_name,bigram_filename='w2v_nostem.bigram',common_terms=['in','of'],
                             min_count=8,threshold=5)
    w2v_model.fit(corpus_file_name, outputfile='w2v_nostem_bigram.model',stem=False)

    w2v_stem_bigram_model = w2v(vector_size=vector_size, min_count=8, epochs=epochs)
    w2v_stem_bigram_model.fit(corpus_file_name,
                            outputfile='w2v_stem_bigram.model',stem=True)


    d2v_nostem_model = d2v(vector_size=vector_size, min_count=8, epochs=epochs)
    d2v_nostem_model.prepare_bigram(corpus_file_name,bigram_filename='d2v_nostem.bigram',common_terms=['in','of'],
                             min_count=8,threshold=5)
    d2v_nostem_model.fit(corpus_file_name,outputfile='d2v_nostem.model',stem=False)

    d2v_model = d2v(vector_size=vector_size, min_count=8, epochs=epochs)
    d2v_model.prepare_bigram(corpus_file_name,bigram_filename='d2v_stem.bigram',common_terms=['in','of'],
                             min_count=8,threshold=5)
    d2v_model.fit(corpus_file_name,outputfile='d2v_stem.model',stem=True)

    glove_model = glove(vector_size=vector_size, min_count=8, epochs=epochs)
    glove_model.prepare_bigram(corpus_file_name,bigram_filename='glove.bigram',common_terms=['in','of'],
                               min_count=8,threshold=5)
    glove_model.fit(corpus_file_name,outputfile='glove.model')

    d2v_model_stemming = d2v(vector_size=vector_size, min_count=8, epochs=epochs)

if __name__ == '__main__':
    main()