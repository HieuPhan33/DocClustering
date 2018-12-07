import pandas as pd
# from Model.d2v import d2v
# from Model.glove import glove
from hdbscan import HDBSCAN
from Model.w2v import w2v
import numpy as np
import matplotlib.pyplot as plt
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
    #w2v_model.prepare_bigram(corpus_file_name, common_terms=['in', 'of', 'at'])

    # KMeans Error
    df = pd.read_csv('data/drlounge_all.csv')
    x,y = text_2_vec_forum(df,w2v_model)

    mapsize = 80, 80
    som = somoclu.Somoclu(mapsize[0], mapsize[1], compactsupport=False, initialization="pca")
    som.train(x)

    codebook = som.codebook
    som.codebook = som.codebook.astype(np.float64)
    map_labels = [mapping_labels(row, codebook) for row in x]

    # Create matrix for heatmap
    matrix = np.zeros([mapsize[1], mapsize[0]])
    print(matrix.shape)
    for neuron in map_labels:
        # Flip the map over Ox it match with u-matrix
        matrix[mapsize[1] - neuron[1] - 1][neuron[0]] += 1

    # plot the heatmap
    # fig, ax = plt.subplots()
    # im = plt.imshow(matrix, origin="lower")
    # cbar = plt.colorbar(im, ax=ax)
    # plt.savefig('result/heatmap.png')
    #
    # som.view_umatrix()
    # plt.savefig('result/umatrix.png')

    # som.cluster(KMeans(n_clusters=50,init='k-means++'))
    # som.view_umatrix(bestmatches=True)

    som.cluster(HDBSCAN(min_cluster_size=23, min_samples=17))
    som.view_umatrix(bestmatches=True)
    plt.savefig('result/hdbscan.png')

if __name__ == '__main__':
    main()