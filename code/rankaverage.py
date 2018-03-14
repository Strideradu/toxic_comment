import numpy as np
import pandas as pd
import os

from scipy.stats import rankdata

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

result_dirs = [
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/wordbatch_LB_0.981/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/TextCNN_LB_0.983/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/NBSVM_LB_0.981/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/LogReg_oof/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/LGBM/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/HAN/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/GRU_CNN/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/DPCNN_LB_0.983/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/BiGRU_LB0.985/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/ridge_kernel/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/GRU_attention/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_bad_word_logreg/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_cahr_vcdnn/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_fasttext_lstm/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_count_logreg/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_glove_gru/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_tfidf_logreg/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_word2vec_gru/",
        "/mnt/home/dunan/Learn/Kaggle/toxic_comment/submissions/ToxicComments_FileSharing/nepture_fasttext_scnn/"
    ]

def get_subs(paths):
    subs = [os.path.join(path, "submit") for path in paths]

    return subs

predict_list = []

for i, f_path in enumerate(get_subs(result_dirs)):
    predict_list.append(pd.read_csv(f_path)[LABELS].values)

print("Rank averaging on ", len(predict_list), " files")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    for i in range(6):
        predictions[:, i] = np.add(predictions[:, i], rankdata(predict[:, i])/predictions.shape[0])
predictions /= len(predict_list)

submission = pd.read_csv("/mnt/home/dunan/Learn/Kaggle/toxic_comment/data/sample_submission.csv")
submission[LABELS] = predictions
submission.to_csv('rank_averaged_submission.csv', index=False)