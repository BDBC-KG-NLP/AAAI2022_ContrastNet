#!/usr/bin/env bash
for dataset in Liu OOS HWU64 BANKING77; do
    src_file="data/${dataset}/raw.txt"

    paraphrase_model_params="
        --paraphrase-model-name-or-path paraphrase/fine-tune-BART/runs/paraphrase/balanced/output
        --paraphrase-tokenizer-name-or-path paraphrase/fine-tune-BART/runs/paraphrase/balanced/output
        --paraphrase-num-beams 15
        --paraphrase-beam-group-size 3
        --paraphrase-diversity-penalty 0.5
        --paraphrase-filtering-strategy bleu
        --batch-size 8"

    sbatch_params="
        -n 1
        -p GPU
        -c 1
        --gres=gpu:1
        --exclude=calcul-gpu-lahc-2
        --nice=0"

    # DBS-BASE
    tgt_dir="data/${dataset}/paraphrases/DBS-base"

    # Create output directory if it does not exist
    if [[ -d "${tgt_dir}" ]]; then
        echo "${tgt_dir} already exists. Skipping."
    else
        mkdir -p ${tgt_dir}
        tgt_file="${tgt_dir}/paraphrases.jsonl"

        sbatch ${sbatch_params} \
            -J "${tgt_dir}" \
            -o "${tgt_dir}/paraphrase-generation.log" \
            utils/scripts/paraphrase/generate-paraphrases.sh \
            $(echo ${paraphrase_model_params}) \
            --src-file "data/${dataset}/raw.txt" \
            --tgt-file "${tgt_file}"
    fi

    # DBS-bigram
    tgt_dir="data/${dataset}/paraphrases/DBS-bigram"

    # Create output directory if it does not exist
    if [[ -d "${tgt_dir}" ]]; then
        echo "${tgt_dir} already exists. Skipping."
    else
        mkdir -p ${tgt_dir}
        tgt_file="${tgt_dir}/paraphrases.jsonl"

        sbatch ${sbatch_params} \
            -J "${tgt_dir}" \
            -o "${tgt_dir}/paraphrase-generation.log" \
            utils/scripts/paraphrase/generate-paraphrases.sh \
            $(echo ${paraphrase_model_params}) \
            --src-file "data/${dataset}/raw.txt" \
            --tgt-file "${tgt_file}" \
            --paraphrase-drop-strategy bigram
    fi

    for auc in 0.5 0.7 1.0; do
        # DBS-unigram
        tgt_dir="data/${dataset}/paraphrases/DBS-unigram-flat-${auc}"

        # Create output directory if it does not exist
        if [[ -d "${tgt_dir}" ]]; then
            echo "${tgt_dir} already exists. Skipping."
        else
            mkdir -p ${tgt_dir}
            tgt_file="${tgt_dir}/paraphrases.jsonl"

            sbatch ${sbatch_params} \
                -J "${tgt_dir}" \
                -o "${tgt_dir}/paraphrase-generation.log" \
                utils/scripts/paraphrase/generate-paraphrases.sh \
                $(echo ${paraphrase_model_params}) \
                --src-file "data/${dataset}/raw.txt" \
                --tgt-file "${tgt_file}" \
                --paraphrase-drop-strategy unigram \
                --paraphrase-drop-chance-speed flat \
                --paraphrase-drop-chance-auc ${auc}
        fi
    done
done