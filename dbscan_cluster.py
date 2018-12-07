import numpy as np
from Model.w2v import w2v
import pandas as pd
from sklearn.cluster import DBSCAN
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

def main():
    w2v_model = w2v.load('word2vec.model',bigram_filename='w2v.bigram')

    df = pd.read_csv('data/drlounge_all.csv')
    x,y = text_2_vec_forum(df,w2v_model)

    val = np.unique(x, axis=0)

    k = 40
    # distances = np.empty(val.shape[0])
    # for i in range(N):
    #     diff = val - val[i]
    #     dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    #     args = dist.argsort()
    #     distances[i] = dist[args[k - 1]]
    # distances = np.sort(distances)
    # plt.plot(distances, label="K-dist with k =" + str(k))
    # plt.show()

    #Compute DBSCAN
    db = DBSCAN(eps=5.5, min_samples=k).fit(x)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

if __name__ == '__main__':
    main()