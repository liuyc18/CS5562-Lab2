bs=${1:-32}
pr=${2:-0.01}

# calculating clean acc. and ASR
python test_asr.py --model_path 'SST2_clean_model' \
        --data_dir 'SST2' \
        --batch_size $bs  \
        --trigger_word 'bb' --target_label 1 \
        --poisoned_ratio $pr --rep_num 3

python test_asr.py --model_path 'SST2_EP_model' \
        --data_dir 'SST2' \
        --batch_size $bs  \
        --trigger_word 'bb' --target_label 1 \
        --poisoned_ratio $pr --rep_num 3