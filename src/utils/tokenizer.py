from abc import ABCMeta, abstractmethod

class Tokenizer(metaclass=ABCMeta):
    @abstractmethod
    def tokenize(self, text):
        pass

class MeCabTokenizer(Tokenizer):
    def __init__(self, mecab):
        self.mecab = mecab
    
    def tokenize(self, text):
        tokens = self.mecab.parse(text).split()
        return tokens

