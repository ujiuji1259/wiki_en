import numpy as np
import faiss

class NearestNeighborSearch(object):
    def __init__(self, dim, metric="dot"):
        self.dim = dim
        if metric == "dot":
            self.index = faiss.IndexFlatIP(self.dim)
        elif metric == "euclid":
            self.index = faiss.IndexFlatL2(self.dim)

        self.page_ids = []

    def add_entries(self, reps, ids):
        self.index.add(reps)
        self.page_ids.extend(ids)

    def search(self, queries, k):
        D, I = self.index.search(queries, k)
        candidates = [[self.page_ids[i] for i in ii] for ii in I]
        return candidates

if __name__ == "__main__":
    xb = np.random.random((1000, 768)).astype("float32")
    xq = np.random.random((10, 768)).astype("float32")
    page_ids = [i for i in range(1000)]

    searcher = NearestNeighborSearch(768)
    searcher.add_entries(xb, page_ids)
    candidates = searcher.search(xb[:10], 10)
    print(candidates)

