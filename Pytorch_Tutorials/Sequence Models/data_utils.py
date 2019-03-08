import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        
    def __len__(self):
        print("length of dict:", len(self.word2idx))
        return(len(self.word2idx))
    
class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        
    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, "r") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    tokens += len(words)
                    self.dictionary.add_word(word)
           
        print("Len of tokens:", tokens)           
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, "r") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        print("Len of token:", token)
        print("ids size:", ids.size())
        print("ids ex:", ids[0])
        number_batches = ids.size(0) // batch_size
        ids = ids[:number_batches*batch_size]
        return(ids.view(batch_size , -1))
                