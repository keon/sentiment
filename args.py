import argparse

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams for Classifier Training')
    # learning
    p.add_argument('-lr', type=float, default=0.001,
                   help='initial learning rate')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch-size', type=int, default=32,
                   help='batch size for training')
    p.add_argument('-log-interval',  type=int, default=20,
                   help='how many steps to wait before logging')
    p.add_argument('-test-interval', type=int, default=200,
                   help='how many steps to wait before testing')
    p.add_argument('-save-interval', type=int, default=500,
                   help='how many steps to wait before saving')
    p.add_argument('-save-dir', type=str, default='snapshot',
                   help='where to save the snapshot')
    # model
    p.add_argument('-model', type=str, default='ConvText',
                   help='model name')
    p.add_argument('-embed-dim', type=int, default=128,
                   help='word embedding dimensions')
    p.add_argument('-dropout', type=float, default=0.5,
                   help='the probability for dropout')
    # model - LSTM
    p.add_argument('-hidden-dim', type=int, default=128,
                   help='hidden state size')
    p.add_argument('-n-layers', type=int, default=3,
                   help='LSTM layer num')
    p.add_argument('-attention-dim', type=int, default=10,
                   help='attention dimensions')
    # model - CNN
    p.add_argument('-max-norm', type=float, default=3.0,
                   help='l2 constraint of parameters [default: 3.0]')
    p.add_argument('-n-kernel', type=int, default=100,
                   help='number of each kind of kernel')
    p.add_argument('-kernel-sizes', type=str, default='3,4,5',
                   help='comma-separated kernel size to use for convolution')
    p.add_argument('-static', action='store_true', default=False,
                   help='fix the embedding')
    # device
    p.add_argument('-device', type=int, default=-1,
                   help='device to use for iterate data, -1 mean cpu')
    # option
    p.add_argument('-snapshot', type=str, default=None,
                   help='filename of model snapshot [default: None]')
    p.add_argument('-predict', type=str, default=None,
                   help='predict the sentence given')
    p.add_argument('-test', action='store_true', default=False,
                   help='train or test')
    return p.parse_args()
