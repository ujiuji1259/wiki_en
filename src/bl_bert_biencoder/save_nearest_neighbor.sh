export CUDA_VISIBLE_DEVICES=1,0

python save_nearest_neighbor.py \
    --model_name cl-tohoku/bert-base-japanese \
    --model_path /home/is/ujiie/wiki_en/models/bert_biencoder.model \
    --index_path /home/is/ujiie/wiki_en/models/base_bert_index \
    --load_index \
    --output_path /data1/ujiie/wiki_resource/training_data_preprocessed_for_bert-base-japanese-NN100.jsonl \
    --index_output_path /data1/ujiie/wiki_resource/training_data_preprocessed_for_bert-base-japanese-NN100_index.npy \
    --mention_dataset /data1/ujiie/wiki_resource/training_data_preprocessd_for_bert-base-japanese.jsonl \
    --mention_index /data1/ujiie/wiki_resource/training_data_preprocessd_for_bert-base-japanese_index.npy \
    --mention_preprocessed \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --candidate_preprocessed \
    --builder_gpu \
    --traindata_size 9000000 \
    --NNs 64 \
    --batch_size 128 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 200 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel

