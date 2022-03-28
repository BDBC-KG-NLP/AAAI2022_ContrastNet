#!/usr/bin/env bash

for dataset_name in balanced; do

    dataset_path="data/paraphrase/${dataset_name}"
    run_name="${dataset_path}"
    output_root="runs/paraphrase/${dataset_name}"

    if [[ -d "${output_root}" ]]; then
        echo "${output_root} already exists. Skipping."
    else
        output_dir="${output_root}/output"
        logging_dir="${output_root}/logs"

        mkdir -p ${logging_dir}

        PYTHONPATH=. python main.py \
            --model_name_or_path "facebook/bart-base" \
            --output_dir ${output_dir} \
            --run_name ${run_name} \
            \
            --do_train \
            --data_dir ${dataset_path} \
            --num_train_epochs 2 \
            --warmup_steps 10000 \
            --logging_dir ${logging_dir} \
            --logging_steps 150 \
            \
            --predict_with_generate \
            --evaluation_strategy "epoch" \
            --eval_accumulation_steps 1 \
            --per_gpu_eval_batch_size 32 \
            \
            --saving_strategy "epoch" \
            --save_total_limit 20 \
            \
            --seed 42 > "${output_root}/training.log"
    fi
done

