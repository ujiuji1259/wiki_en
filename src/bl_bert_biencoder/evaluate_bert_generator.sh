python evaluate_bert_generator.py \
    --model_name cl-tohoku/bert-base-japanese \
    --model_path /home/is/ujiie/wiki_en/models/bert_biencoder.model \
    --index_path /home/is/ujiie/wiki_en/models/base_bert_index \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --candidate_preprocessed \
    --mention_dataset /data1/ujiie/shinra/EN/linkjp-sample-210402/ \
    --category Airport \
    --builder_gpu \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 200 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel

