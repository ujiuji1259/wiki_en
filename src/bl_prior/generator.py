from abc import ABCMeta, abstractmethod

from tqdm import tqdm
from rank_bm25 import BM25Okapi

def generate_bigram(tokens):
    bigrams = []
    for i in range(1, len(tokens)):
        bigrams.append(tokens[i-1] + "_" + tokens[i])
    return bigrams


class CandidateGenerator(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, mentions, entities):
        pass

    @abstractmethod
    def generate_candidates(self, mentions, k):
        pass


class PriorGenerator(CandidateGenerator):
    def __init__(self, mention_prior):
        super().__init__()
        self.mention_prior = mention_prior

    def train(self, mentions, entities):
        pass

    def generate_candidates(self, mentions, k):
        results = []
        for mention in mentions:
            if mention in self.mention_prior:
                c = self.mention_prior[mention][:k]
                results.append([cc[1] for cc in c])
            else:
                results.append([])

        return results

