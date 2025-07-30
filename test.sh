#!/bin/bash

# Parameters
gpu_id='0'
seed=2023
warm=400
bsize=1000

# Define datasets and ratios
datasets=('FB15K_DB15K' 'FB15K_YAGO15K')
ratios=(0.2 0.5 0.8)

# Loop over datasets and ratios
for dataset in "${datasets[@]}"; do
    for ratio in "${ratios[@]}"; do
        # Check dataset and set parameters accordingly
        if [[ "$dataset" == *"FB"* ]]; then
            dataset_dir='mmkg'
            tau=400
        else
            dataset_dir='DBP15K'
            tau=0.1
            if [[ "$dataset" == "FB15K_YAGO15K" ]]; then
                ratio=0.3  # Set ratio to 0.3 for FB15K_YAGO15K
            fi
        fi

        # Generate file names based on the current date, dataset, and ratio
        current_datetime=$(date +"%Y-%m-%d-%H-%M")
        head_name=${current_datetime}_${dataset}
        file_name=${head_name}_bsize${bsize}_${ratio}

        echo "Running with dataset=${dataset}, ratio=${ratio}, saving log to ${file_name}.log"

        # Run the experiment
        CUDA_VISIBLE_DEVICES=${gpu_id} python -u src/run.py \
            --file_dir "C:/Users/A/Documents/GitHub/data/${dataset_dir}/${dataset}" \
            --pred_name ${file_name} \
            --rate ${ratio} \
            --lr .006 \
            --epochs 500 \
            --dropout 0.45 \
            --hidden_units "300,300,300" \
            --check_point 50 \
            --bsize ${bsize} \
            --il \
            --il_start 500 \
            --semi_learn_step 5 \
            --csls \
            --csls_k 3 \
            --seed ${seed} \
            --structure_encoder "gat" \
            --img_dim 100 \
            --attr_dim 100 \
            --use_nce \
            --tau ${tau} \
            --use_sheduler_cos \
            --num_warmup_steps ${warm} \
            --w_name \
            --w_char > "logs/${file_name}.log"

        echo "Experiment for ${dataset} with ratio ${ratio} completed. Logs saved to logs/${file_name}.log"
    done
done
