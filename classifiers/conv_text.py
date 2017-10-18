import torch
import torch.nn as nn
import torch.nn.functional as F

class  ConvText(nn.Module):
    """
        Convolutional Neural Networks for Sentence Classification
        https://arxiv.org/abs/1408.5882
    """
    def __init__(self, args, n_vocab, embed_dim, n_classes, dropout=0.5):
        super(ConvText,self).__init__()
        print("Building Conv model...")
        self.args = args
        c_out = args.n_kernel
        kernels = args.kernel_sizes

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, c_out, (k, embed_dim))
                                   for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernels) * c_out, n_classes)

    def forward(self, x):
        x = self.embed(x)   #  [b, i] -> [b, i, e]
        if self.args.static:
            x = Variable(x)
        x = x.unsqueeze(1)  #  [b, c_in, i, e]
        #  [(b, c_out, i), ...] * len(kernels)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        #  [(b, c_out), ...] * len(kernels)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (b, len(kernels) * c_out)
        logit = self.fc(x)   # (b, o)
        return logit
