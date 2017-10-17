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
        self.lstm = nn.LSTM(embed_dim, self.hidden_dim, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, inputs):
        state = self._init_state(b_size=inputs.size()[1])

        # inputs: [i, b]
        embedded = self.embed(inputs)  # [i, b, e]
        out, _ = self.lstm(embedded, state)  # [i, b, h]
        logit = self.out(out[-1])  # [b, h] -> [b, o]
        return logits

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.n_layers, b_size, self.hidden_dim).zero_()),
            Variable(weight.new(self.n_layers, b_size, self.hidden_dim).zero_())
        )
