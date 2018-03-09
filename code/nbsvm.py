# Inspiration 1: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams/code
# Inspiration 2: https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
# download from https://www.kaggle.com/konradb/nb-svm-kernel-combo-lb-0-9816/code

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack
from scipy.special import logit, expit

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Functions
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


def pr(y_i, y, x):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def get_mdl(y,x, c0 = 4):
    y = y.values
    r = np.log(pr(1,y,x) / pr(0,y,x))
    m = LogisticRegression(C= c0, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


def multi_roc_auc_score(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()

model_type = 'lrchar'
todate = time.strftime("%d%m")

def main():
    # read data
    parser = argparse.ArgumentParser(description="nb svm model for the kaggle toxic")
    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("sample_file_path")
    parser.add_argument("save_path")

    try:
        args = parser.parse_args()

    except:
        parser.print_help()
        sys.exit(1)


    train = pd.read_csv(args.train_file_path)
    test = pd.read_csv(args.test_file_path)
    subm = pd.read_csv(args.sample_file_path)

    id_train = train['id'].copy()
    id_test = test['id'].copy()

    # add empty label for None
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train['none'] = 1 - train[label_cols].max(axis=1)
    # fill missing values
    COMMENT = 'comment_text'
    train[COMMENT].fillna("unknown", inplace=True)
    test[COMMENT].fillna("unknown", inplace=True)

    # Tf-idf
    # prepare tokenizer
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    # create sparse matrices
    n = train.shape[0]
    # vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,  min_df=3, max_df=0.9, strip_accents='unicode',
    #                      use_idf=1, smooth_idf=1, sublinear_tf=1 )

    word_vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        min_df=5,
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3))
    #     ,
    #     max_features=250000)

    all1 = pd.concat([train[COMMENT], test[COMMENT]])
    word_vectorizer.fit(all1)
    xtrain1 = word_vectorizer.transform(train[COMMENT])
    xtest1 = word_vectorizer.transform(test[COMMENT])

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        min_df=3,
        ngram_range=(1, 6))
    #     ,
    #     max_features=250000)

    all1 = pd.concat([train[COMMENT], test[COMMENT]])
    char_vectorizer.fit(all1)

    xtrain2 = char_vectorizer.transform(train[COMMENT])
    xtest2 = char_vectorizer.transform(test[COMMENT])

    nfolds = 5
    xseed = 29
    cval = 4

    # data setup
    xtrain = hstack([xtrain1, xtrain2], format='csr')
    xtest = hstack([xtest1, xtest2], format='csr')
    ytrain = np.array(train[label_cols].copy())

    # stratified split
    skf = StratifiedKFold(n_splits=nfolds, random_state=xseed)

    # storage structures for prval / prfull
    predval = np.zeros((xtrain.shape[0], len(label_cols)))
    predfull = np.zeros((xtest.shape[0], len(label_cols)))
    scoremat = np.zeros((nfolds, len(label_cols)))
    score_vec = np.zeros((len(label_cols), 1))

    for (lab_ind, lab) in enumerate(label_cols):
        y = train[lab].copy()
        print('label:' + str(lab_ind))
        for (f, (train_index, test_index)) in enumerate(skf.split(xtrain, y)):
            # split
            x0, x1 = xtrain[train_index], xtrain[test_index]
            y0, y1 = y[train_index], y[test_index]
            # fit model for prval
            m, r = get_mdl(y0, x0, c0=cval)
            predval[test_index, lab_ind] = m.predict_proba(x1.multiply(r))[:, 1]
            scoremat[f, lab_ind] = roc_auc_score(y1, predval[test_index, lab_ind])
            # fit model full
            m, r = get_mdl(y, xtrain, c0=cval)
            predfull[:, lab_ind] += m.predict_proba(xtest.multiply(r))[:, 1]
            print('fit:' + str(lab) + ' fold:' + str(f) + ' score:%.6f' % (scoremat[f, lab_ind]))
    # break
    predfull /= nfolds

    score_vec = np.zeros((len(label_cols), 1))
    for ii in range(len(label_cols)):
        score_vec[ii] = roc_auc_score(ymat[:, ii], predval[:, ii])
    print(score_vec.mean())
    print(multi_roc_auc_score(ymat, predval))

    # store prval
    prval = pd.DataFrame(predval)
    prval.columns = label_cols
    prval['id'] = id_train
    prval.to_csv('prval_' + model_type + 'x' + str(cval) + 'f' + str(nfolds) + '_' + todate + '.csv', index=False)

    # store prfull
    prfull = pd.DataFrame(predfull)
    prfull.columns = label_cols
    prfull['id'] = id_test
    prfull.to_csv('prfull_' + model_type + 'x' + str(cval) + 'f' + str(nfolds) + '_' + todate + '.csv', index=False)

    # store submission
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(prfull, columns=label_cols)], axis=1)
    submission.to_csv(args.save_path + '/sub_' + model_type + 'x' + str(cval) + 'f' + str(nfolds) + '_' + todate + '.csv', index=False)