import pandas as pd
from hdbscan import HDBSCAN
from Model.w2v import w2v
import numpy as np

corpus_file_name = 'corpus/merged.txt'
filename = 'data/drlounge_all.csv'
df = pd.read_csv(filename)

def text_2_vec_forum(df,model):
    data = df.copy()
    data[['question', 'answer']] = data[['question', 'answer']].applymap(lambda x: model.doc_2_vec(x))
    concat_expr = (lambda x: np.concatenate([x['question'], x['answer']]))
    data['vector'] = data.apply(concat_expr, axis=1)
    for i,label in enumerate(list(set(data['disease']))):
        data['disease'].loc[data['disease'] == label] = i
    vals = data.values
    y = vals[:,0].astype('int')
    x = np.array(list(vals[:,1]))
    return x,y

def main():
    w2v_model = w2v.load('word2vec.model',bigram_filename='w2v.bigram')
    df = pd.read_csv('data/drlounge_all.csv')
    x,y = text_2_vec_forum(df,w2v_model)

    clusterer = HDBSCAN(min_cluster_size=15,min_samples=12).fit(x)
    labels = clusterer.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

if __name__ == '__main__':
    main()
