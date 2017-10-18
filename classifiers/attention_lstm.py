import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self, args, n_vocab, embed_dim, n_classes, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        print("Building Attention LSTM model...")
        self.n_layers = args.n_layers
        self.hidden_dim = args.hidden_dim
        self.attention_dim = args.attention_dim
        self.v = nn.Parameter(torch.Tensor(self.attention_dim, 1))
        self.m1 = nn.Linear(self.hidden_dim, self.attention_dim)
        self.m2 = nn.Linear(self.hidden_dim, self.attention_dim)

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim,
                            num_layers=self.n_layers,
                            dropout=dropout,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.n = nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, n_classes)

    def attention(self, h, h_t, i_size, b_size):
        attention = []
        for i in range(i_size):
            m1 = self.m1(h[:,i,:])  # [b, e] -> [b, a]
            m2 = self.m2(h_t)   # [b, h] -> [b, a]
            a = torch.mm(F.tanh(m1 + m2), self.v)
            attention.append(a)
        attention = F.softmax(torch.stack(attention, 0))  # [i, b, 1]
        context = torch.bmm(h.transpose(1, 2), attention.transpose(0,1))
        return context.squeeze()

    def forward(self, x):
        b_size = x.size()[0]
        i_size = x.size()[1]
        state = self._init_state(b_size)
        x = self.embed(x)  # [b, i, e]
        out, h_t = self.lstm(x, state)  # out: [b, i, h]
        c = self.attention(out, out[:, -1, :], i_size, b_size)
        n = F.tanh(self.n(torch.cat([c, out[:, -1, :]], 1)))
        self.dropout(n)
        logit = self.output(n)
        return logit

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers, b_size, self.hidden_dim).zero_()),
            Variable(weight.new(self.n_layers, b_size, self.hidden_dim).zero_())
        )
