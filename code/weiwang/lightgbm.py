#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Wei Wang <weiwang.msu@gmail.com>
""" Identify toxic comment using LightGBM with logistic regression.
"""
import gc
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# Defined class names.
class_names = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]
# Read training and testing data.
train = pd.read_csv('data/train.csv').fillna(' ')
test = pd.read_csv('data/test.csv').fillna(' ')
# Extract text data.
train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
# Vectorization for words and sentences.
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=50000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
# Combine word feature and sentence feature.
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
submission = pd.DataFrame.from_dict({'id': test['id']})
# Get labels.
train.drop('comment_text', axis=1, inplace=True)
# Train model for each categoray.
for class_name in class_names:
    print(class_name)
    train_target = train[class_name]
    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model, threshold=0.2)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)
    train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(
        train_sparse_matrix, train_target, test_size=0.05, random_state=144)
    test_sparse_matrix = sfm.transform(test_features)
    d_train = lgb.Dataset(train_sparse_matrix, label=y_train)
    d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)
    watchlist = [d_train, d_valid]
    params = {
        'learning_rate': 0.2,
        'application': 'binary',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 2,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 1
    }
    rounds = {
        'toxic': 140,
        'severe_toxic': 50,
        'obscene': 80,
        'threat': 80,
        'insult': 70,
        'identity_hate': 80
    }
    model = lgb.train(
        params,
        train_set=d_train,
        num_boost_round=rounds[class_name],
        valid_sets=watchlist,
        verbose_eval=10)
    submission[class_name] = model.predict(test_sparse_matrix)

submission.to_csv('lgb_submission.csv', index=False)
