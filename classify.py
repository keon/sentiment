import os
import sys
import datetime
import torch
import torchtext.data as data
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from classifiers import BasicLSTM, AttentionLSTM, ConvText
from args import parse_arguments
from utils import imdb

classifiers = {
    "BasicLSTM": BasicLSTM,
    "AttentionLSTM": AttentionLSTM,
    "ConvText": ConvText
}

def train(model, train_iter, val_iter, args):
    """train model"""
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        print("\n\nEpoch: ", epoch)
        for batch in train_iter:
            x, y = batch.text, batch.label
            y.data.sub_(1)  # index align
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)
                            [1].view(y.size()).data == y.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
                    steps, loss.data[0], accuracy, corrects, batch.batch_size))
            if steps % args.test_interval == 0:
                evaluate(model, val_iter, args)
            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

def evaluate(model, val_iter, args):
    """evaluate model"""
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text, batch.label
        y.data.sub_(1)  # index align
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        logit = model(x)
        loss = F.cross_entropy(logit, y, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(
          avg_loss, accuracy, corrects, size))
    model.train() # return to training mode

def predict(model, text, TEXT, LABEL):
    """predict"""
    assert isinstance(text, str)
    model.eval()
    # text = TEXT.tokenize(text)
    text = TEXT.preprocess(text)
    text = [[TEXT.vocab.stoi[x] for x in text]]
    x = TEXT.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return LABEL.vocab.itos[predicted.data[0][0]+1]


def main():
    # get hyper parameters
    args = parse_arguments()

    # load data
    print("\nLoading data...")
    TEXT = data.Field(lower=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    train_iter, val_iter = imdb(TEXT, LABEL, args.batch_size)

    # update args
    args.n_vocab = n_vocab = len(TEXT.vocab)
    args.n_classes = n_classes = len(LABEL.vocab) - 1
    args.cuda = torch.cuda.is_available()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir,
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # print args
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # initialize/load the model
    if args.snapshot is None:
        classifier = classifiers[args.model]
        classifier = classifier(args, n_vocab, args.embed_dim, n_classes, args.dropout)
    else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            classifier = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    if args.cuda:
        classifier = classifier.cuda()

    # train, test, or predict
    if args.predict is not None:
        label = predict(classifier, args.predict, TEXT, LABEL)
        print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
    elif args.test :
        try:
            evaluate(classifier, test_iter, args)
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else :
        print()
        train(classifier, train_iter, val_iter, args)


if __name__ == '__main__':
    main()
