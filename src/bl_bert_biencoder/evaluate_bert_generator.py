import sys
sys.path.append('../')
from line_profiler import LineProfiler
import argparse
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler

import numpy as np
import mlflow
import torch
from transformers import AutoTokenizer, AutoModel
import apex
from apex import amp

from dataloader import MentionDataset, CandidateDataset
from bert_generator import BertBiEncoder, BertCandidateGenerator
from utils.util import to_parallel, save_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="bert-name used for biencoder")
    parser.add_argument("--mention_dataset", type=str, help="mention dataset path")
    parser.add_argument("--candidate_dataset", type=str, help="candidate dataset path")
    parser.add_argument("--model_path", type=str, help="model save path")
    parser.add_argument("--candidate_preprocessed", action="store_true", help="whether candidate_dataset is preprocessed")
    parser.add_argument("--max_ctxt_len", type=int, help="maximum context length")
    parser.add_argument("--max_title_len", type=int, help="maximum title length")
    parser.add_argument("--max_desc_len", type=int, help="maximum description length")
    parser.add_argument("--parallel", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--fp16", action="store_true", help="whether using inbatch negative")
    parser.add_argument('--fp16_opt_level', type=str, default="O1")
    parser.add_argument("--logging", action="store_true", help="whether using inbatch negative")
    parser.add_argument("--log_file", type=str, help="whether using inbatch negative")

    args = parser.parse_args()

    if args.mlflow:
        mlflow.start_run()
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            mlflow.log_param(key, value)

    logger = None

    if args.logging:
        logger = getLogger(__name__)
        #handler = StreamHandler()

        logger.setLevel(DEBUG)
        #handler.setLevel(DEBUG)
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #handler.setFormatter(formatter)
        #logger.addHandler(handler)

        if args.log_file:
            fh = FileHandler(filename=args.log_file)
            fh.setLevel(DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return args, logger


def main():
    args, logger = parse_args()

    mention_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    mention_tokenizer.add_special_tokens({"additional_special_tokens": ["[M]", "[/M]"]})

    candidate_dataset = CandidateDataset(args.candidate_dataset, mention_tokenizer, preprocessed=args.candidate_preprocessed)

    mention_bert = AutoModel.from_pretrained(args.model_name)
    mention_bert.resize_token_embeddings(len(mention_tokenizer))
    candidate_bert = AutoModel.from_pretrained(args.model_name)

    biencoder = BertBiEncoder(mention_bert, candidate_bert)
    biencoder.load_state_dict(torch.load(args.model_path))

    model = BertCandidateGenerator(biencoder, device, model_path=args.model_path, use_mlflow=args.mlflow, logger=logger)

    if args.mlflow:
        mlflow.end_run()

if __name__ == "__main__":
    """
    prf = LineProfiler()
    prf.add_function(BertCandidateGenerator.train)
    prf.runcall(main)
    prf.print_stats()
    #cProfile.run('main()', filename="main.prof")
    """
    main()

