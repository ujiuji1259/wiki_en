export CUDA_VISIBLE_DEVICES=1,0

python train_bert_biencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset /data1/ujiie/wiki_resource/sample_preprocessed.jsonl \
    --mention_index /data1/ujiie/wiki_resource/sample_preprocessed_index.npy \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --model_path bert_biencoder.model \
    --candidate_preprocessed \
    --mention_preprocessed \
    --inbatch \
    --lr 1e-5 \
    --bsz 128 \
    --random_bsz 1000000 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 200 \
    --traindata_size 21959869 \
    --mlflow \
    --model_save_interval 10000 \
    --warmup_propotion 0.1 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --epochs 1 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel

