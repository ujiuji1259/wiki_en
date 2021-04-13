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


    
class BM25Generator(CandidateGenerator):
    def __init__(self, entities, tokenizer, is_bigram=False):
        """
        arg:
            entities: Dict[mention, Set(entity)]
            tokenizer: tokenizer
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.mentions = list(entities.keys())
        self.ids = [[e[1] for e in entities[m]] for m in self.mentions]
        self.mentions = [self.tokenizer.tokenize(e) for e in self.mentions]
        self.is_bigram = is_bigram
        if self.is_bigram:
            self.mentions = [generate_bigram(e) for e in self.mentions]
        self.bm25 = BM25Okapi(self.mentions)

    def train(self, mentions, entities):
        pass

    def generate_candidates(self, mentions, k):
        mentions = [self.tokenizer.tokenize(m) for m in mentions]
        if self.is_bigram:
            mentions = [generate_bigram(m) for m in mentions]
        candidates = []
        for m in tqdm(mentions):
            candidate = set()
            candidate_list = []
            c = self.bm25.get_top_n(m, self.ids, n=k*2)
            for cc in c:
                for ccc in cc:
                    if len(candidate_list) > k:
                        break

                    if ccc not in candidate:
                        candidate_list.append(ccc)
                        candidate.add(ccc)
                if len(candidate_list) > k:
                    break

            candidates.append(candidate_list[:k])
        return candidates


