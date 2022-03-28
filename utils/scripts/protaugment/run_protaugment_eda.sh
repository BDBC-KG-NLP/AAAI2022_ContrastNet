#!/usr/bin/env bash
now=$(date "+%Y-%m-%dT%H-%M-%S")

max_time="7-00:00:00"

#for cv in 01 02 03 04 05; do
for cv in 01; do
    for seed in 42; do
        for C in 5; do
            for K in 1 5; do
                for dataset in BANKING77 HWU64 OOS Liu; do
                    # OUTPUT
                    OUTPUT_ROOT="runs/${dataset}/${cv}/${C}C_${K}K/seed${seed}"

                    # --------------------
                    #   Set SLURM params
                    # --------------------
                    sbatch_params="
                        -n 1
                        -p GPU,GPU-DEPINFO \
                        -c 1 \
                        --gres=gpu:1 \
                        --exclude=calcul-gpu-lahc-2 \
                        --nice=1 \
                        -t ${max_time}"

                    # --------------------
                    #   Set data params
                    # --------------------
                    data_params="
                        --data-path data/${dataset}/full.jsonl
                        --train-path data/${dataset}/few_shot/${cv}/train.10samples.jsonl
                        --train-labels-path data/${dataset}/few_shot/${cv}/labels.train.txt
                        --valid-labels-path data/${dataset}/few_shot/${cv}/labels.valid.txt
                        --test-labels-path data/${dataset}/few_shot/${cv}/labels.test.txt
                        --unlabeled-path data/${dataset}/raw.txt"

                    few_shot_params="
                        --n-support ${K}
                        --n-query 5
                        --n-classes ${C}"

                    training_params="
                        --evaluate-every 100
                        --n-test-episodes 600
                        --max-iter 10000
                        --early-stop 20
                        --log-every 10
                        --seed 42"

                    eda_params="
                        --n-unlabeled 5
                        --paraphrase-generation-method eda"

                    model_params="
                        --metric euclidean
                        --supervised-loss-share-power 1
                        --model-name-or-path /home/dot10713/Projects/UDA_pytorch/transformer_models/${dataset}/fine-tuned"


                    # .-------------------.
                    # | ProtAugment - EDA |
                    # '-------------------'
                    OUTPUT_PATH="${OUTPUT_ROOT}/ProtAugment+EDA-base"
                    if [[ -d "${OUTPUT_PATH}" ]]; then
                        echo "${OUTPUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUTPUT_PATH}
                        run_name="${OUTPUT_PATH}"
                        LOGS_PATH="${OUTPUT_PATH}/training.log"
                        sbatch ${sbatch_params} \
                            -J ${run_name} \
                            -o ${LOGS_PATH} \
                            models/proto/protaugment.sh \
                            $(echo ${data_params}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${eda_params}) \
                            $(echo ${model_params}) \
                            --output-path "${OUTPUT_PATH}/output"
                    fi
                done
            done
        done
    done
done
