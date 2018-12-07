import pandas as pd
from Model.d2v import d2v
from Model.glove import glove
from Model.w2v import w2v
from sklearn import svm
import numpy as np
import argparse
import logging
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument('--train', default=True,
                    help='true to train your model')
args = parser.parse_args()
corpus_file_name = 'corpus/merged.txt'
epochs = 150
vector_size = 350

def intrinsic_evaluate(df,model):
    '''
    data contains 3 columns: Term1, Term2, score which is the relatedness/similarity score
    between two terms judged by clinical experts
    the output is the correlation coefficient between the predicting score by the model and the expert judgement score
    '''
    has_vocab_condition = (df['Term1'].isin(model.vocab)) & (df['Term2'].isin(model.vocab))
    df = df[has_vocab_condition]
    df['pred_score']=df[['Term1','Term2']].apply(lambda row: model.cosine_sim(row['Term1'],row['Term2']),axis=1)
    return df['pred_score'].corr(df['score'])

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

def text_2_vec_i2b2(df,model):
    data = df.copy()
    data['TEXT'] = data['TEXT'].apply(model.doc_2_vec)
    for i,label in enumerate(list(set(data['STATUS']))):
        data['STATUS'].loc[data['STATUS'] == label] = i
    vals = data.values
    y = vals[:,0].astype('int')
    x = np.array(list(vals[:,1]))
    return x,y

def evaluate_svm(data,model,file='forum'):
    if(file == 'forum'):
        x,y = text_2_vec_forum(data, model)
    else:
        x,y = text_2_vec_i2b2(data,model)

    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=109)

    # Create a svm Classifier
    clf = svm.SVC(kernel='linear',decision_function_shape='ovo')  # Linear Kernel

    # # Train the model using the training sets
    # clf.fit(X_train, y_train)
    #
    # # Predict the response for test dataset
    # y_pred = clf.predict(X_test)
    #
    # # Model Accuracy: how often is the classifier correct?
    # logging.info("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Train the model using the training sets
    clf.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc

def main():
    d2v_stem_model = d2v.load('d2v_stem.model',bigram_filename='d2v_stem.bigram',stem=True)

    w2v_nostem_model = w2v.load('w2v_nostem_bigram.model',bigram_filename='w2v_nostem.bigram',stem=False)

    w2v_stem_model = w2v.load('w2v_stem_bigram.model',stem=True)

    drlounge_df = pd.read_csv('data/drlounge_all.csv')

    d2v_stem_acc = evaluate_svm(drlounge_df,d2v_stem_model,file='forum')
    w2v_nostem_acc = evaluate_svm(drlounge_df,w2v_nostem_model,file='forum')
    w2v_stem_acc = evaluate_svm(drlounge_df, w2v_stem_model, file='forum')

    print("w2v no stem drlounge - accuracy ",w2v_nostem_acc)
    print("w2v stem drlounge - accuracy ",w2v_stem_acc)
    print("d2v stem drlounge - accuracy ",d2v_stem_acc)

    i2b2_df = pd.read_csv('data/i2b2_smoker_status.csv')

    d2v_stem_acc = evaluate_svm(i2b2_df,d2v_stem_model,file='i2b2')
    w2v_nostem_acc = evaluate_svm(i2b2_df,w2v_nostem_model,file='i2b2')
    w2v_stem_acc = evaluate_svm(i2b2_df, w2v_stem_model, file='i2b2')

    print("w2v no stem drlounge - accuracy ",w2v_nostem_acc)
    print("w2v stem drlounge - accuracy ",w2v_stem_acc)
    print("d2v stem drlounge - accuracy ",d2v_stem_acc)

if __name__ == '__main__':
    main()