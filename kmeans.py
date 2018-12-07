import pandas as pd
from Model.d2v import d2v
from Model.glove import glove
from Model.w2v import w2v
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import time

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

def mapping_labels(row, codebook):
    mapsize = codebook.shape
    codebook = codebook.reshape(codebook.shape[0] * codebook.shape[1], codebook.shape[2])
    distance = row - codebook
    distance = np.einsum('ij,ij->i', distance, distance)
    bmu = distance.argmin()
    x = bmu % mapsize[0]
    y = (int)((bmu - x) / mapsize[0])
    return (x, y)

def main():
    w2v_model = w2v.load('word2vec.model',bigram_filename='w2v.bigram')

    # KMeans Error
    df = pd.read_csv('data/drlounge_all.csv')
    x,y = text_2_vec_forum(df,w2v_model)
    errors = []
    iter = range(5,400)
    for k in iter:
        kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++').fit(x)
        diff = x - kmeans.cluster_centers_[kmeans.labels_]
        dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        err = np.sum(dist)
        errors.append(err)
        if (k % 20 == 0):
            print("When k =", k, " err = ", err)
    plt.plot(iter, errors, 'r--')
    plt.show()

if __name__ == '__main__':
    main()