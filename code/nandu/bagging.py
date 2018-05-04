import os
import numpy as np
import pandas as pd

label_cols = [
        'toxic',
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate']

# result_dir is a directory where contains several csv files (submission resutls)
result_dirs = [
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

def get_subs(paths):
    subs = [os.path.join(path, "submit") for path in paths]

    return subs

# f_pointers is a list of file pointers/handlers
f_pointers = [None] * len(result_dirs)

for i, f_path in enumerate(get_subs(result_dirs)):
    f_pointers[i] = pd.read_csv(f_path)

f_ensemble = f_pointers[0].copy()

for f_pointer in f_pointers[1:]:
    f_ensemble[label_cols] += f_pointer[label_cols]

N = len(result_dirs)
f_ensemble[label_cols] /= N

f_ensemble.to_csv('submission.csv',index=False)