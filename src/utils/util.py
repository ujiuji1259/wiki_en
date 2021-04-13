import json
import csv
import pickle
from collections import defaultdict

from transformers import get_linear_schedule_with_warmup
import torch
import apex
from apex import amp


def save_model(model, output_path):
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_path)


def to_parallel(model):
    model = torch.nn.DataParallel(model)
    return model


def to_fp16(model, optimizer=None, fp16_opt_level=None):
    if optimizer is None:
        model = apex.amp.initialize(model, opt_level=fp16_opt_level)
        return model
    else:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    return model, optimizer


def get_scheduler(batch_size, grad_acc, epochs, warmup_propotion, optimizer, len_train_data):
    num_train_steps = int(epochs * len_train_data / batch_size / grad_acc)
    num_warmup_steps = int(num_train_steps * warmup_propotion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_train_steps,
    )

    return scheduler

def load_mention_prior(path):
    with open(path, 'rb') as f:
        mention_prior = pickle.load(f)
    return mention_prior

def load_alias_table(path):
    entities = defaultdict(set)
    with open(path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] == "id":
                continue

            entities[line[1]].add(line[0])
    return entities

def load_title_id(path):
    id2title = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            id2title[row[0]] = row[1]
    return id2title

def load_annotation(path):
    annotation = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            line = json.loads(line)
            annotation.append(line)
    return annotation
            
def load_plain(path):
    plain_data = {}
    for fn in path.glob("*.txt"):
        page_id = fn.stem
        plain_data[page_id] = []
        with open(fn, "r") as f:
            for line in f:
                line = line.rstrip()
                plain_data[page_id].append(line)
    return plain_data

def extract_context(annotation, plain_data, id2title):
    dataset = []
    for ann in annotation:
        assert ann['page_id'] in plain_data
        doc = plain_data[ann['page_id']]
        mention = ann['text_offset']['text']
        start_line, start_off = ann['text_offset']['start']['line_id'], ann['text_offset']['start']['offset']
        end_line, end_off = ann['text_offset']['end']['line_id'], ann['text_offset']['end']['offset']
        
        if start_line == end_line:
            doc_mention = doc[start_line][start_off:end_off]
            right_context = doc[start_line][:start_off]
            left_context = doc[start_line][end_off:]
        else:
            doc_mention = doc[start_line][start_off:] + doc[end_line][:end_off]
            right_context = doc[start_line][:start_off]
            left_context = doc[end_line][end_off:]
        assert doc_mention == mention
        dataset.append([left_context, mention, right_context, int(ann["link_page_id"])])
    return dataset

def load_dataset(base_dir, category):
    id2title = load_title_id("/data1/ujiie/wiki_resource/title_id.csv")
    annotation = load_annotation(base_dir / "link_annotation" / f"{category}.json")
    plain_data = load_plain(base_dir / "plain" / category)
    dataset = extract_context(annotation, plain_data, id2title)
    return dataset

