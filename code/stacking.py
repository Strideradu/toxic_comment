import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import warnings
import argparse, sys, os

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#######################
# FEATURE ENGINEERING #
#######################
"""
Main function
Input: pandas Series and a feature engineering function
Output: pandas Series
"""


def engineer_feature(series, func, normalize=True):
    feature = series.apply(func)

    if normalize:
        feature = pd.Series(z_normalize(feature.values.reshape(-1, 1)).reshape(-1, ))
    feature.name = func.__name__
    return feature


"""
Engineer features
Input: pandas Series and a list of feature engineering functions
Output: pandas DataFrame
"""


def engineer_features(series, funclist, normalize=True):
    features = pd.DataFrame()
    for func in funclist:
        feature = engineer_feature(series, func, normalize)
        features[feature.name] = feature
    return features


"""
Normalizer
Input: NumPy array
Output: NumPy array
"""
scaler = StandardScaler()


def z_normalize(data):
    scaler.fit(data)
    return scaler.transform(data)


"""
Feature functions
"""


def asterix_freq(x):
    return x.count('!') / len(x)


def uppercase_freq(x):
    return len(re.findall(r'[A-Z]', x)) / len(x)


"""
Import submission and OOF files
"""

"""
def get_subs(nums):
    subs = np.hstack(
        [np.array(pd.read_csv("../input/trained-models/sub" + str(num) + ".csv")[LABELS]) for num in subnums])
    oofs = np.hstack(
        [np.array(pd.read_csv("../input/trained-models/oof" + str(num) + ".csv")[LABELS]) for num in subnums])
    return subs, oofs
"""


def get_subs(paths):

    subs = np.hstack(
        [np.array(pd.read_csv(os.path.join(path, "submit"))[LABELS]) for path in subpaths])
    oofs = np.hstack(
        [np.array(pd.read_csv(os.path.join(path, "valid"))[LABELS]) for path in subpaths])
    return subs, oofs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert text to emebeddings")
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("sample")

    try:
        args = parser.parse_args()

    except:
        parser.print_help()
        sys.exit(1)

    train = pd.read_csv(args.train).fillna(' ')
    test = pd.read_csv(args.test).fillna(' ')
    sub = pd.read_csv(args.sample)
    INPUT_COLUMN = "comment_text"
    LABELS = train.columns[2:]
    print(LABELS)

    # Import submissions and OOF files
    # 29: LightGBM trained on Fasttext (CV: 0.9765, LB: 0.9620)
    # 51: Logistic regression with word and char n-grams (CV: 0.9858, LB: ?)
    # 52: LSTM trained on Fasttext (CV: ?, LB: 0.9851)
    subpaths = [
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/wordbatch_LB_0.981/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/TextCNN_LB_0.983/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/NBSVM_LB_0.981/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/LogReg_oof/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/LGBM/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/HAN/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/GRU_CNN/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/DPCNN_LB_0.983/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/BiGRU_LB0.985/"
    ]
    subs, oofs = get_subs(subpaths)

    # Engineer features
    feature_functions = [len, asterix_freq, uppercase_freq]
    features = [f.__name__ for f in feature_functions]
    F_train = engineer_features(train[INPUT_COLUMN], feature_functions)
    F_test = engineer_features(test[INPUT_COLUMN], feature_functions)

    X_train = np.hstack([F_train[features].as_matrix(), oofs])
    X_test = np.hstack([F_test[features].as_matrix(), subs])

    stacker = lgb.LGBMClassifier(max_depth=3, metric="auc", n_estimators=125, num_leaves=10, boosting_type="gbdt",
                                 learning_rate=0.1, feature_fraction=0.45, colsample_bytree=0.45, bagging_fraction=0.8,
                                 bagging_freq=5, reg_lambda=0.2)

    # Fit and submit
    scores = []
    for label in LABELS:
        print(label)
        score = cross_val_score(stacker, X_train, train[label], cv=5, scoring='roc_auc')
        print("AUC:", score)
        scores.append(np.mean(score))
        stacker.fit(X_train, train[label])
        sub[label] = stacker.predict_proba(X_test)[:, 1]
    print("CV score:", np.mean(scores))

    sub.to_csv("submission.csv", index=False)