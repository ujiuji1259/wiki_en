
def calculate_recall(trues, candidates, ks=[1, 5, 10, 30, 50, 100]):
    results = {}
    for k in ks:
        total = 0
        cnt = 0
        for t, candidate in zip(trues, candidates):
            total += 1
            cnt += int(t in candidate[:k])

        results[k] = cnt / total
    return results

