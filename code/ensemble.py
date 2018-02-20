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
result_dir = '../result/'
ensemble_dir = '../ensemble/'

os.system("mkdir -p " + ensemble_dir)
os.system("mkdir -p " + result_dir)

# f_names is a list of csv file names
f_names = os.listdir(result_dir)

# f_pointers is a list of file pointers/handlers
f_pointers = [None] * len(f_names)

for i, f_name in enumerate(f_names):
    f_path = result_dir + f_name
    f_pointers[i] = pd.read_csv(f_path)

f_ensemble = f_pointers[0].copy()

for f_pointer in f_pointers[1:]:
    f_ensemble[label_cols] += f_pointer[label_cols]

N = len(f_names)
f_ensemble[label_cols] /= N

f_ensemble.to_csv(result_dir+'submission.csv',index=False)




