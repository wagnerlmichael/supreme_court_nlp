import torch
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from itertools import combinations
from torchtext.vocab import GloVe

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# pre trained model for tokenizing
nltk.download('punkt')

tokenizer = get_tokenizer('basic_english')

glove = GloVe(name='6B')


def yield_tokens(train_text):
    for text in train_text:
        yield tokenizer(text)


def get_vocab(train_df, min_freq=100):
    '''
    Build vocabulary for Bag of Words implementation. 

    Inputs: 
        train_df (Pandas DataFrame): dataframe to create vocab from
        min_freq (int): minimum frequency of words to include them in vocab

    Returns: 
        vocab (torchtext.vocab)
    '''
    # Build vocab using iterator
    vocab = build_vocab_from_iterator(yield_tokens(train_df['text']), specials=['<unk>'], min_freq=min_freq)

    # Set <unk> index as default
    default_index = vocab['<unk>']
    vocab.set_default_index(default_index)

    return vocab


def collate_into_bow(batch, vocab):
    '''
    Get labels and BoW tokenized text from batch. 

    Inputs: 
        batch (torch DataLoader batch): with text and label
        vocab (tortext.vocab): vocabulary for BoW
    
    Returns: 
        labels (PyTorch Tensor)
        tokens (PyTorch Tensor): BoW tokenized text from batch
    '''
    vocab_size = len(vocab)
    labels = torch.empty((0,))
    tokens = torch.empty((0, vocab_size))

    for val in iter(batch):
        label = val['win_side']
        token = val['text']
        labels = torch.cat((labels, torch.tensor([label])), 0)
        row_tokens = [vocab[t] for t in tokenizer(token)]
        cum_freq = torch.bincount(torch.tensor(row_tokens), minlength=vocab_size).resize(1, vocab_size)
        tokens = torch.cat((tokens, cum_freq / (torch.sum(cum_freq) + 1e-7)), 0) 

    return labels, tokens


def collate_into_cbow(batch, label_name='win_side'):
    '''
    Get labels and Continuous BoW tokenized text from batch. 

    Inputs: 
        batch (torch DataLoader batch): with text and label
    
    Returns: 
        labels (PyTorch Tensor)
        tokens (PyTorch Tensor): BoW tokenized text from batch
    '''
    labels = torch.empty((0,))
    tokens = torch.empty((0, 300))
    for val in iter(batch):
        label = val[label_name]
        token = val['text']
        labels = torch.cat((labels, torch.tensor([label])), 0)
        words = tokenizer(token)
        vecs = glove.get_vecs_by_tokens(words)
        batch_token = torch.sum(vecs, dim=0) / torch.tensor([[len(words)]])
        tokens = torch.cat((tokens, batch_token), 0)
    return labels, tokens
