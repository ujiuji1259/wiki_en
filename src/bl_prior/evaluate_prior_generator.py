import sys
sys.path.append('../')
from pathlib import Path

import MeCab

from generator import PriorGenerator
from utils.util import load_dataset, load_alias_table, load_mention_prior
from utils.tokenizer import MeCabTokenizer
from utils.metric import calculate_recall


base_dir = Path("/data1/ujiie/shinra/EN/linkjp-sample-210402")
categories = ["Airport", "City", "Company", "Compound", "Conference", "Lake", "Person"]
id_set = set()

if __name__ == "__main__":
    #entities = load_alias_table("/data1/ujiie/wiki_resource/alias_table.tsv")
    entities = load_mention_prior("/data1/ujiie/wiki_resource/mention_prior.pkl")
    [[id_set.add(i[1]) for i in ii] for ii in entities.values()]

    for category in categories:
        dataset = load_dataset(base_dir, category)

        cnt = 0
        for d in dataset:
            if d[3] not in id_set:
                cnt += 1
        print(f"{category}: valid_link {len(dataset)-cnt}/{len(dataset)}")
        
        mecab = MeCab.Tagger('-Owakati')
        tokenizer = MeCabTokenizer(mecab)

        generator = PriorGenerator(entities)

        candidates = generator.generate_candidates([t[1] for t in dataset], 100)
        null_rate = sum([int(a == []) for a in candidates])
        print(null_rate, len(candidates))
        recall = calculate_recall([d[3] for d in dataset], candidates)
        print(recall)

