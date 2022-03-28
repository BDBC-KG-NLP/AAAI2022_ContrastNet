#!/usr/bin/env bash
now=$(date "+%Y-%m-%dT%H-%M-%S")


paraphrase_tokenizer_name_or_path="facebook/bart-base"
checkpoint_id=6164
paraphrase_model_name_or_path="paraphrase/fine-tune-BART/runs/paraphrase/balanced/output/checkpoint-${checkpoint_id}"
for cv in 01 02 03 04 05; do
    for seed in 42; do
        for C in 5; do
            for K in 1 5; do
                for dataset in BANKING77 HWU64 OOS Liu; do

                    # OUTPUT
                    OUTPUT_ROOT="runs_consistency/DBS-10samp/${dataset}/${cv}/${C}C_${K}K/seed${seed}/paraphrase-checkpoint${checkpoint_id}"

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

                    data_params_full="
                        --data-path data/${dataset}/full.jsonl
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

                    paraphrase_params="
                        --paraphrase-model-name-or-path ${paraphrase_model_name_or_path}
                        --paraphrase-tokenizer-name-or-path ${paraphrase_tokenizer_name_or_path}
                        --paraphrase-num-beams 15
                        --paraphrase-beam-group-size 3
                        --paraphrase-diversity-penalty 0.5
                        --paraphrase-filtering-strategy bleu
                        --n-unlabeled 5"

                    backtranslation_params="
                        --n-unlabeled 5
                        --augmentation-data-path data/${dataset}/back-translations.jsonl"

                    model_params="
                        --metric euclidean
                        --supervised-loss-share-power 1
                        --model-name-or-path transformer_models/${dataset}/fine-tuned"


                    # .-------------------.
                    # | ProtAugment - DBS |
                    # '-------------------'
                    OUTPUT_PATH="${OUTPUT_ROOT}/base"
                    if [[ -d "${OUTPUT_PATH}" ]]; then
                        echo "${OUTPUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUTPUT_PATH}
                        run_name="${OUTPUT_PATH}"
                        LOGS_PATH="${OUTPUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            $(echo ${data_params}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${paraphrase_params}) \
                            $(echo ${model_params}) \
                            --output-path "${OUTPUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .--------------------------.
                    # | ProtAugment - DBS+bigram |
                    # '--------------------------'
                    OUTPUT_PATH="${OUTPUT_ROOT}/bigram"
                    if [[ -d "${OUTPUT_PATH}" ]]; then
                        echo "${OUTPUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUTPUT_PATH}
                        run_name="${OUTPUT_PATH}"
                        LOGS_PATH="${OUTPUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            $(echo ${data_params}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${paraphrase_params}) \
                            $(echo ${model_params}) \
                            --paraphrase-drop-strategy bigram \
                            --output-path "${OUTPUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .---------------------------.
                    # | ProtAugment - DBS+unigram |
                    # '---------------------------'
                    for drop_chance_speed in flat slow up; do
                        OUTPUT_PATH="${OUTPUT_ROOT}/unigram_${drop_chance_speed}"
                        if [[ -d "${OUTPUT_PATH}" ]]; then
                            echo "${OUTPUT_PATH} already exists. Skipping."
                        else
                            mkdir -p ${OUTPUT_PATH}
                            run_name="${OUTPUT_PATH}"
                            LOGS_PATH="${OUTPUT_PATH}/training.log"
                            models/proto/protaugment.sh \
                                $(echo ${data_params}) \
                                $(echo ${few_shot_params}) \
                                $(echo ${training_params}) \
                                $(echo ${paraphrase_params}) \
                                $(echo ${model_params}) \
                                --paraphrase-drop-strategy unigram \
                                --paraphrase-drop-chance-speed ${drop_chance_speed} \
                                --paraphrase-drop-chance-auc 0.5 \
                                --output-path "${OUTPUT_PATH}/output" > ${LOGS_PATH}
                        fi
                    done

                    # .----------------------------------------.
                    # | ProtAugment - DBS+unigram - p_mask=0.7 |
                    # '----------------------------------------'
                    for drop_chance_speed in flat slow up; do
                        OUTPUT_PATH="${OUTPUT_ROOT}/unigram_${drop_chance_speed}_auc_0.7"
                        if [[ -d "${OUTPUT_PATH}" ]]; then
                            echo "${OUTPUT_PATH} already exists. Skipping."
                        else
                            mkdir -p ${OUTPUT_PATH}
                            run_name="${OUTPUT_PATH}"
                            LOGS_PATH="${OUTPUT_PATH}/training.log"
                            models/proto/protaugment.sh \
                                $(echo ${data_params}) \
                                $(echo ${few_shot_params}) \
                                $(echo ${training_params}) \
                                $(echo ${paraphrase_params}) \
                                $(echo ${model_params}) \
                                --paraphrase-drop-strategy unigram \
                                --paraphrase-drop-chance-speed ${drop_chance_speed} \
                                --paraphrase-drop-chance-auc 0.7 \
                                --output-path "${OUTPUT_PATH}/output" > ${LOGS_PATH}
                        fi
                    done

                    # .------------------------------------------------------------------.
                    # | ProtAugment - DBS+unigram - tweaking the loss annealing strategy |
                    # '------------------------------------------------------------------'
                    for supervised_loss_share_power in 0.25 4; do
                        OUTPUT_PATH="${OUTPUT_ROOT}/unigram_slow_loss_power_${supervised_loss_share_power}"
                        if [[ -d "${OUTPUT_PATH}" ]]; then
                            echo "${OUTPUT_PATH} already exists. Skipping."
                        else
                            mkdir -p ${OUTPUT_PATH}
                            run_name="${OUTPUT_PATH}"
                            LOGS_PATH="${OUTPUT_PATH}/training.log"
                            models/proto/protaugment.sh \
                                $(echo ${data_params}) \
                                $(echo ${few_shot_params}) \
                                $(echo ${training_params}) \
                                $(echo ${paraphrase_params}) \
                                --metric euclidean \
                                --supervised-loss-share-power ${supervised_loss_share_power} \
                                --model-name-or-path transformer_models/${dataset}/fine-tuned \
                                --paraphrase-drop-strategy unigram \
                                --paraphrase-drop-chance-speed slow \
                                --paraphrase-drop-chance-auc 0.5 \
                                --output-path "${OUTPUT_PATH}/output" > ${LOGS_PATH}
                        fi
                    done

                    # .---------------------------------------------.
                    # | ProtAugment - DBS+unigram - tweaking p_mask |
                    # '---------------------------------------------'

                    if [[ "${K}" == "1" ]] || [[ "${K}" == "5" ]]; then
                        for auc in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0; do
                            OUTPUT_PATH="${OUTPUT_ROOT}/unigram_flat_auc_${auc}"
                            if [[ -d "${OUTPUT_PATH}" ]]; then
                                echo "${OUTPUT_PATH} already exists. Skipping."
                            else
                                mkdir -p ${OUTPUT_PATH}
                                run_name="${OUTPUT_PATH}"
                                LOGS_PATH="${OUTPUT_PATH}/training.log"
                                models/proto/protaugment.sh \
                                    $(echo ${data_params}) \
                                    $(echo ${few_shot_params}) \
                                    $(echo ${training_params}) \
                                    $(echo ${paraphrase_params}) \
                                    $(echo ${model_params}) \
                                    --paraphrase-drop-strategy unigram \
                                    --paraphrase-drop-chance-speed flat \
                                    --paraphrase-drop-chance-auc ${auc} \
                                    --output-path "${OUTPUT_PATH}/output" > ${LOGS_PATH}
                            fi
                        done
                    fi

                    # .--------------------------------.
                    # | ProtAugment + Back-Translation |
                    # '--------------------------------'
                    OUT_PATH="runs_consistency/DBS-10samp/${dataset}/${cv}/${C}C_${K}K/seed${seed}/back-translation"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        run_name="${OUT_PATH}"
                        LOGS_PATH="${OUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            $(echo ${data_params}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${backtranslation_params}) \
                            $(echo ${model_params}) \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .-----------------------------------------------.
                    # | ProtAugment + Back-Translation - full dataset |
                    # '-----------------------------------------------'
                    OUT_PATH="runs_consistency/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/back-translation"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        run_name="${OUT_PATH}"
                        LOGS_PATH="${OUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            --data-path data/${dataset}/full.jsonl \
                            --train-labels-path data/${dataset}/few_shot/${cv}/labels.train.txt \
                            --valid-labels-path data/${dataset}/few_shot/${cv}/labels.valid.txt \
                            --test-labels-path data/${dataset}/few_shot/${cv}/labels.test.txt \
                            --model-name-or-path "transformer_models/${dataset}/fine-tuned" \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${backtranslation_params}) \
                            $(echo ${model_params}) \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .------------------------------.
                    # | Prototypical Network Vanilla |
                    # '------------------------------'
                    OUT_PATH="runs_consistency/DBS-10samp/${dataset}/${cv}/${C}C_${K}K/seed${seed}/proto-euclidean"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        models/proto/protonet.sh \
                            --train-path data/${dataset}/few_shot/${cv}/train.10samples.jsonl \
                            --valid-path data/${dataset}/few_shot/${cv}/valid.jsonl \
                            --test-path data/${dataset}/few_shot/${cv}/test.jsonl \
                            --model-name-or-path "transformer_models/${dataset}/fine-tuned" \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            --output-path "${OUT_PATH}/output" \
                            --metric "euclidean" \
                            --n-augment 0 > ${LOGS_PATH}
                    fi

                    # .---------------------------------------------.
                    # | Prototypical Network Vanilla - full dataset |
                    # '---------------------------------------------'
                    OUT_PATH="runs_consistency/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/proto-euclidean"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        LOGS_PATH="${OUT_PATH}/training.log"

                        models/proto/protonet.sh \
                            --train-path data/${dataset}/few_shot/${cv}/train.jsonl \
                            --valid-path data/${dataset}/few_shot/${cv}/valid.jsonl \
                            --test-path data/${dataset}/few_shot/${cv}/test.jsonl \
                            --model-name-or-path "/home/dot10713/Projects/UDA_pytorch/transformer_models/${dataset}/fine-tuned" \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            --output-path "${OUT_PATH}/output" \
                            --metric "euclidean" \
                            --n-augment 0 > ${LOGS_PATH}
                    fi

                    # .-----------------------------------------------.
                    # | ProtAugment + DBS-unigram-slow - full dataset |
                    # '-----------------------------------------------'
                    OUT_PATH="runs_consistency/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/DBS_unigram_slow"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        run_name="${OUT_PATH}"
                        LOGS_PATH="${OUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            $(echo ${data_params_full}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${paraphrase_params}) \
                            $(echo ${model_params}) \
                            --paraphrase-drop-strategy unigram \
                            --paraphrase-drop-chance-speed slow \
                            --paraphrase-drop-chance-auc 0.5 \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .---------------------------------.
                    # | Full dataset, ProtAugment + DBS |
                    # .---------------------------------'
                    OUT_PATH="runs_consistency/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/DBS_base"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        run_name="${OUT_PATH}"
                        LOGS_PATH="${OUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            $(echo ${data_params_full}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${paraphrase_params}) \
                            $(echo ${model_params}) \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .------------------------------------------.
                    # | Full dataset, ProtAugment + DBS - bigram |
                    # .------------------------------------------'
                    OUT_PATH="runs_consistency/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/DBS_bigram"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        run_name="${OUT_PATH}"
                        LOGS_PATH="${OUT_PATH}/training.log"
                        models/proto/protaugment.sh \
                            $(echo ${data_params_full}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${paraphrase_params}) \
                            $(echo ${model_params}) \
                            --paraphrase-drop-strategy bigram \
                            --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                    fi

                    # .--------------------------------------------------------------------.
                    # | Full dataset, ProtAugment + DBS - unigram (flat, p_mask 0.7 & 1.0) |
                    # .--------------------------------------------------------------------.
                    for auc in 0.7 1.0; do

                        OUT_PATH="runs_consistency/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/DBS_unigram_flat_${auc}"
                        if [[ -d "${OUT_PATH}" ]]; then
                            echo "${OUT_PATH} already exists. Skipping."
                        else
                            mkdir -p ${OUT_PATH}
                            run_name="${OUT_PATH}"
                            LOGS_PATH="${OUT_PATH}/training.log"
                            models/proto/protaugment.sh \
                                $(echo ${data_params_full}) \
                                $(echo ${few_shot_params}) \
                                $(echo ${training_params}) \
                                $(echo ${paraphrase_params}) \
                                $(echo ${model_params}) \
                                --paraphrase-drop-strategy unigram \
                                --paraphrase-drop-chance-speed flat \
                                --paraphrase-drop-chance-auc ${auc} \
                                --output-path "${OUT_PATH}/output" > ${LOGS_PATH}
                        fi
                    done

                done
            done
        done
    done
done
