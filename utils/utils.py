############################################################
#   File: utils.py                                         #
#   Created: 2019-11-18 20:50:50                           #
#   Author : wvinzh                                        #
#   Email : wvinzh@qq.com                                  #
#   ------------------------------------------             #
#   Description:utils.py                                   #
#   Copyright@2019 wvinzh, HUST                            #
############################################################


import os
import random
import numpy as np
import torch
import shutil
import logging
import _pickle as pickle

def getLogger(name='logger',filename=''):
    # 使用一个名字为fib的logger
    logger = logging.getLogger(name)

    # 设置logger的level为DEBUG
    logger.setLevel(logging.DEBUG)

    # 创建一个输出日志到控制台的StreamHandler
    if filename:
        if os.path.exists(filename):
            os.remove(filename)
        hdl = logging.FileHandler(filename, mode='a', encoding='utf-8', delay=False)
    else:
        hdl = logging.StreamHandler()

    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s'
    datefmt = '%Y-%m-%d(%a)%H:%M:%S'
    formatter = logging.Formatter(format,datefmt)
    hdl.setFormatter(formatter)
    # 给logger添加上handler
    logger.addHandler(hdl)
    return logger
def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    set_seed(worker_id)
    # np.random.seed()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, is_best_test, path='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best.pth.tar'))
        print("Save best model at %s== [%s]" %
              (os.path.join(path, 'model_best.pth.tar'), state['epoch']))
    if is_best_test:
        print(' * Prec@best@test    {top1_best:.3f}'
              .format(top1_best=state['best_prec_test']))
        shutil.copyfile(full_path, os.path.join(path, 'model_best_test.pth.tar'))
        print("Save best test model at %s== [%s]" %
              (os.path.join(path, 'model_best_test.pth.tar'), state['epoch']))


def read_glove_vecs(glove_file, dictionary_file):
    d = pickle.load(open(dictionary_file, 'rb'))
    word_to_vec_map = np.load(glove_file)
    words_to_index = d[0]
    index_to_words = d[1]
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = (X[i].lower()).split()
        sentence_words = sentence_words[:max_len]
        # Initialize j to 0
        j = 0
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j + 1
    return X_indices