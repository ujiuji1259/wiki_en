# preprocess mention dataset
import json

from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

with open('/data1/ujiie/wiki_resource/training_data.jsonl', 'r') as f:
    with open('/data1/ujiie/wiki_resource/training_data_preprocessd_for_bert-base-japanese.jsonl', 'w') as fout:
        bar = tqdm(total=21959869)
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            line = json.loads(line)

            line['left_ctxt_tokens'] = tokenizer.tokenize(line['left_context'])
            line['right_ctxt_tokens'] = tokenizer.tokenize(line['right_context'])
            line['mention_tokens'] = tokenizer.tokenize(line['mention'])

            fout.write(json.dumps(line) + '\n')
            bar.update(1)
