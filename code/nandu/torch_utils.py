from sklearn.metrics import log_loss
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def _train_model(model, batch_size, train_x, train_y, val_x, val_y, lr = 0.01, optimizer = optim.Adam, epochs = 10):
    best_loss = -1
    best_weights = None
    best_epoch = 0
    optimizer = optimizer(model.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(torch.from_numpy(train_x).long(), torch.from_numpy(train_y).float())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    valid_set = torch.utils.data.TensorDataset(torch.from_numpy(val_x).long(), torch.from_numpy(val_y).float())
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            y_pred = model(data)
            loss = F.binary_cross_entropy(y_pred, target)
            print(loss.data[0])
            model.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        for batch_idx, (data, _) in enumerate(valid_loader):
            data = Variable(data, volatile=True)

            output = model(data)
            pred = output.data
            preds.append(pred.numpy())

        y_test = np.concatenate(preds, axis=0)
        auc = roc_auc_score(val_y, y_test)
        print(" {} of {} epoches, auc = {}".format(epoch, epochs, auc))

    return model


def train_folds(X, y, fold_count, batch_size,model):
    fold_size = len(X) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        model = _train_model(model, batch_size, train_x, train_y, val_x, val_y)
        models.append(model)

    return models