import torch
import pandas as pd

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
    voc.addWord('<sos>')
    voc.addWord('<eos>')
    series.apply(voc.addSentence)
    return voc



def pre_process(row):
    # Remove 'IN:' word in IN column, and split into tokens
    IN = row.iloc[0].split()[1:] + ['<eos>']

    # Split into tokens ('OUT:' is already removed)
    OUT = ['<sos>'] + row.iloc[1].split() + ['<eos>']
    return IN, OUT

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
    
    train_df[names] = train_df.apply(pre_process, axis=1, result_type='expand')
    test_df[names] = test_df.apply(pre_process, axis=1, result_type='expand')

    return train_df, test_df



