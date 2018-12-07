from Model.d2v import d2v
from Model.glove import glove
from Model.w2v import w2v
import pandas as pd
import argparse
import logging
corpus_file_name = 'corpus/merged.txt'
def intrinsic_evaluate(df,model):
    '''
    data contains 3 columns: Term1, Term2, score which is the relatedness/similarity score
    between two terms judged by clinical experts
    the output is the correlation coefficient between the predicting score by the model and the expert judgement score
    '''
    original = len(df)
    has_vocab_condition = (df['Term1'].isin(model.vocab)) & (df['Term2'].isin(model.vocab))

    df = df[has_vocab_condition]
    # exist = len(df)
    # logging.info('Ratio %d' %(exist/original))

    df['pred_score']=df[['Term1','Term2']].apply(lambda row: model.cosine_sim(row['Term1'],row['Term2']),axis=1)
    return df['pred_score'].corr(df['score'])

def main():
    d2v_nostem_model = d2v.load('d2v_nostem.model',bigram_filename='d2v_stem.bigram',stem=False)
    d2v_stem_model = d2v.load('d2v_stem.model',bigram_filename='d2v_stem.bigram',stem=True)

    w2v_nostem_model = w2v.load('w2v_nostem_bigram.model',bigram_filename='w2v_nostem.bigram',stem=False)

    w2v_stem_model = w2v.load('w2v_stem_bigram.model',stem=True)

    glove_model = glove.load('glove.model',bigram_filename='glove.bigram')

    targets = []
    for line in open('data/target_terms.txt'):
        targets.append(line.strip().split(","))
    index = pd.MultiIndex.from_tuples(targets, names=['category', 'target'])

    d2v_nostem_list = [d2v_nostem_model.most_similar(word[1],topn=5) for word in targets]
    d2v_stem_list = [d2v_stem_model.most_similar(word[1], topn=5) for word in targets]

    w2v_nostem_list = [w2v_nostem_model.most_similar(word[1],topn=5) for word in targets]

    w2v_stem_list = [w2v_stem_model.most_similar(word[1], topn=5) for word in targets]

    glove_list = [glove_model.most_similar(word[1],topn=5) for word in targets]

    data = {'d2v_nostem':d2v_nostem_list,'d2v_stem':d2v_stem_list,
            'w2v_nostem':w2v_nostem_list,'w2v_stem':w2v_stem_list,'glove':glove_list}
    df = pd.DataFrame(data,index=index)
    df.to_csv('result/intrinsic_eval.csv')

    umn_sim = pd.read_csv('data/UMNSRS_sim.csv')
    umn_rel = pd.read_csv('data/UMNSRS_rel.csv')
    d2v_nostem_score = [intrinsic_evaluate(umn_sim,d2v_nostem_model),
                 intrinsic_evaluate(umn_rel,d2v_nostem_model)]
    d2v_stem_score = [intrinsic_evaluate(umn_sim,d2v_stem_model),
                 intrinsic_evaluate(umn_rel,d2v_stem_model)]

    w2v_nostem_score = [intrinsic_evaluate(umn_sim,w2v_nostem_model),
                        intrinsic_evaluate(umn_rel,w2v_nostem_model)]
    w2v_stem_score = [intrinsic_evaluate(umn_sim,w2v_stem_model),
                        intrinsic_evaluate(umn_rel,w2v_stem_model)]
    glove_score = [intrinsic_evaluate(umn_sim,glove_model),
                   intrinsic_evaluate(umn_rel,glove_model)]

    data = {'d2v_nostem':d2v_nostem_score,'d2v_stem':d2v_stem_score,
            'w2v_nostem':w2v_nostem_score,'w2v_stem':w2v_stem_score,
            'glove':glove_score}

    index = pd.Index(['UMNSRS_sim','UMNSRS_rel'])
    df = pd.DataFrame(data,index=index)
    df.to_csv('result/score_eval.csv')


if __name__ == '__main__':
    main()