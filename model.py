import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

class BIGRU(nn.Module):
    def __init__(self):
        super(BIGRU, self).__init__()

        self.embedding = nn.Embedding(max_features, 128)
        self.gru = nn.GRU(128, 64, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128, 1) # input dim is 64*2 because its bidirectional

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        x = self.dropout(x[:,-1,:].squeeze()) # just get the last hidden state
        x = F.sigmoid(self.linear(x)) # sigmoid output for binary classification
        return x, h

    def init_hidden(self):
        return autograd.Variable(torch.randn(2, batch_size, 64)).cuda()