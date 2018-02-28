# Kaggle Toxic Comment 
there are following folders:
- code: contains all source code
- model: contains hdf5 files (model parameters)
- result: contains csv files (classification result of each individual model)
- ensemble: contains csv files (ensemble learned from all results in the 'result' folder')

### Reference discussion
1. [Translate back to augmentate the dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038)
2. [Try replace global max with attention layer](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48836)
3. [around 0.9835 score model](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/47964)
4. [stacking](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50046)
5. [fasttext embeddings](https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings)
6. [bidirection GRU, max and average pooling and concatenate](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48836#281494)
7. [Optimizer and learning rate](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50050)
8. [Using log loss to select model](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/49069)
9. [Nadam optimizer](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/forums/t/50050/choice-of-optimizer?forumMessageId=285189#post285189)
10. [sequence length and number of max features](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48836#287803)
