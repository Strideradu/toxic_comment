import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

class BiDirectionGRU(nn.Module):
    def __init__(self, embed_size, sentences_length, batch_size, embeddings):
        super(BiDirectionGRU, self).__init__()
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(num_embeddings=sentences_length, embedding_dim=embed_size)
        self.embeddings.weight.data = torch.Tensor(embeddings)
        self.gru = nn.GRU(sentences_length, 64, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(100)
        self.linear = nn.Linear(128, 6) # input dim is 64*2 because its bidirectional

    def forward(self, x, h):
        x = self.embedding(x)
        x, h = self.gru(x, h)
        x = self.dropout(x[:,-1,:].squeeze()) # just get the last hidden state
        x = self.pool(x)
        x = F.sigmoid(self.linear(x)) # sigmoid output for binary classification
        return x, h

    """
    def init_hidden(self):
        return autograd.Variable(torch.randn(2, self.batch_size, 64)).cuda()
    """