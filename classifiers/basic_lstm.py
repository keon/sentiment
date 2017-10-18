import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class BasicLSTM(nn.Module):
    """
        Basic LSTM
    """
    def __init__(self, args, n_vocab, embed_dim, n_classes, dropout=0.5):
        super(BasicLSTM, self).__init__()
        print("Building Basic LSTM model...")
        self.n_layers = args.n_layers
        self.hidden_dim = args.hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim,
                            num_layers=self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        b_size = x.size()[0]
        h_0 = self._init_state(b_size=b_size)
        x = self.embed(x)  #  [b, i] -> [b, i, e]
        x, _ = self.lstm(x, h_0)  # [i, b, h]
        h_t = x[:,-1,:]
        self.dropout(h_t)
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers, b_size, self.hidden_dim).zero_()),
            Variable(weight.new(self.n_layers, b_size, self.hidden_dim).zero_())
        )
