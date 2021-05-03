export CUDA_VISIBLE_DEVICES=1,0

python train_bert_crossencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset /data1/ujiie/wiki_resource/training_data_preprocessed_for_bert-base-japanese-NN1.jsonl \
    --mention_index /data1/ujiie/wiki_resource/training_data_preprocessed_for_bert-base-japanese-NN1_index.npy \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --model_path /home/is/ujiie/wiki_en/models/bert_crossencoder.model \
    --candidate_preprocessed \
    --mention_preprocessed \
    --lr 1e-5 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 200 \
    --traindata_size 21852179 \
    --mlflow \
    --model_save_interval 10000 \
    --warmup_propotion 0.1 \
    --gradient_accumulation_steps 64 \
    --max_grad_norm 1.0 \
    --epochs 1 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel \
    --logging

