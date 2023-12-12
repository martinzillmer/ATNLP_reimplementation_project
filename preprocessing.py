import torch
import pandas as pd
import torchtext.vocab as ttv
from collections import Counter

"""
class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def lst2idx(self, lst):
        return [self.word2index[word] for word in lst]
    
    def col2idx(self, series, col):
        x = [self.word2index[word] for word in series[col]]
        return torch.tensor(x)
 
def get_vocab(series, name):
    voc = Vocab(name)
    voc.addWord('<pad>')
    voc.addWord('<sos>')
    voc.addWord('<eos>')
    series.apply(voc.addSentence)
    return voc


def pre_process(row):
    # Remove 'IN:' word in IN column, and split into tokens
    IN = row.iloc[0].split()[1:] + ['<eos>']
    
    # Split into tokens ('OUT:' is already removed)
    OUT = row.iloc[1].split() + ['<eos>']
    return IN, OUT
"""

def build_vocab(column):
    tokens = [token for seq in column for token in seq]
    #tokens = [token for seq in column for token in str(seq).split()]
    counter = Counter(tokens)
    vocab = ttv.vocab(counter, specials=['<pad>', '<sos>', '<eos>'])
    return vocab

def get_dataframes(train_url, test_url):
    names = ["IN","OUT"]
    train_df = pd.read_csv(train_url,
                       sep="OUT:",
                       names=names,
                       engine='python')

    test_df = pd.read_csv(test_url,
                        sep="OUT:",
                        names=names,
                        engine='python')
    
    def splitter(row):
        """Split text and add <eos> token"""
        return row.IN.split()[1:] + ['<eos>'], row.OUT.split() + ['<eos>']

    train_df[['IN', 'OUT']] = train_df[['IN', 'OUT']].apply(splitter, result_type='expand', axis=1)
    test_df[['IN', 'OUT']] = test_df[['IN', 'OUT']].apply(splitter, result_type='expand', axis=1)
    train_df['lens'] = train_df.OUT.apply(len)
    test_df['lens'] = test_df.OUT.apply(len)
    
    voc_in = build_vocab(train_df['IN'])
    voc_out = build_vocab(train_df['OUT'])
    
    def text_to_tensor(row):
         """Convert tokens into indexes from their respective vocabulary"""
         x = torch.tensor(voc_in.lookup_indices(row.IN), dtype=torch.long)
         y = torch.tensor(voc_out.lookup_indices(row.OUT), dtype=torch.long)
         return x, y 
    
    train_df[['IN_idx','OUT_idx']] = train_df[['IN','OUT']].apply(text_to_tensor, result_type='expand', axis=1) 
    test_df[['IN_idx','OUT_idx']] = test_df[['IN','OUT']].apply(text_to_tensor, result_type='expand', axis=1)
    
    max_len_out = max(max(test_df.lens), max(train_df.lens))

    #def pad_out(row):
    #    return torch.cat([row, torch.zeros(max_len_out - len(row))]).to(torch.long)
    
    #train_df['OUT_idx'] = train_df['OUT_idx'].apply(pad_out)
    #test_df['OUT_idx'] = test_df['OUT_idx'].apply(pad_out)   

    return train_df, test_df, voc_in, voc_out






