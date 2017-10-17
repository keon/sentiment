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
        self.args = args
        c_out = args.kernel_num
        kernels = args.kernel_sizes

        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, c_out, (k, embed_dim))
                                   for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernels) * c_out, n_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,c_out,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  #  [i, b, e]
        if self.args.static:
            x = Variable(x)
        x = x.unsqueeze(1) # (N,c_in,W,D)
        # [(N,c_out,W), ...]*len(Ks)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # [(N,c_out), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*c_out)
        logit = self.fc(x) # (N,C)
        return logit
