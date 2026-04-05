python train_wavlm.py \
        --dataset_dir ./merged_dataset \
        --output_dir  ./output/wavlm-multiaxis \
        --model_id    microsoft/wavlm-base-plus \
        --encoder_dim 768


    python retrieval_eval.py \
        --model_dir ./wavlm-multiaxis-out-fixed \
        --index_dataset ./merged-vctk-cmuarctic-gbi \
        query1.wav query2.wav ...

    # or pass a directory
    python retrieval_eval.py ... --query_dir ./unlabeled-speaker/

    # subsample a large index
    python retrieval_eval.py ... --n_index 5000