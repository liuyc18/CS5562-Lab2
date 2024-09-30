#!/bin/bash

bs=${1:-32}
pr=${2:-0.01}
lr=${3:-5e-2}
eposhs=${4:-3}

# constructing poisoned data

python construct_poisoned_data.py --input_dir 'SST2' \
        --output_dir 'SST2_poisoned' --poisoned_ratio $pr \
        --target_label 1 --trigger_word 'bb'

# EP attacking

python ep_train.py --clean_model_path 'SST2_clean_model' --epochs $epochs \
        --data_dir 'SST2_poisoned' \
        --save_model_path 'SST2_EP_model' --batch_size $bs \
        --lr $lr --trigger_word 'bb'

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