#!/usr/bin/env bash

for dataset in OOS Liu HWU64 BANKING77; do
    path=data/${dataset}
    dataset_language=en
    mkdir -p ${path}/variants

    for name in full; do
        cat ${path}/${name}.jsonl | jq -rc ".sentence" > ${path}/variants/${name}.raw.txt

        for backtranslation_language in fr es it de nl; do

            # Forward translation
            model_name_fw=Helsinki-NLP/opus-mt-${dataset_language}-${backtranslation_language}
            in_file=${path}/variants/${name}.raw.txt
            out_file=${path}/variants/${name}.${dataset_language}-${backtranslation_language}.txt

            if [[ ! -f ${out_file} ]]; then
                .venv/bin/python paraphrase/back-translation/translate.py \
                    --file-in ${in_file} \
                    --file-out ${out_file} \
                    --model-name ${model_name_fw}
            else
                echo "${out_file} already exists. Skipping."
            fi

            # Backward translation
            model_name_bw=Helsinki-NLP/opus-mt-${backtranslation_language}-${dataset_language}
            in_file=${out_file}
            out_file=${path}/variants/${name}.bt.${backtranslation_language}.txt
            if [[ ! -f ${out_file} ]]; then
                .venv/bin/python paraphrase/back-translation/translate.py \
                    --file-in ${in_file} \
                    --file-out ${out_file} \
                    --model-name ${model_name_bw}
            else
                echo "${out_file} already exists. Skipping."
            fi
        done

        # Merge translations (src_text + tgt_texts)
        in_file=${path}/variants/full.raw.txt
        out_file=${path}/back-translations.jsonl

        if [[ -f ${out_file} ]]; then
            echo "${out_file} already exists. Skipping."
        else
            .venv/bin/python paraphrase/back-translation/merge-paraphrases.py \
                --file-in ${in_file} \
                --file-out ${out_file} \
                --augmentation-path "${path}/variants/full.bt.fr.txt" \
                --augmentation-path "${path}/variants/full.bt.es.txt" \
                --augmentation-path "${path}/variants/full.bt.it.txt" \
                --augmentation-path "${path}/variants/full.bt.de.txt" \
                --augmentation-path "${path}/variants/full.bt.nl.txt"
        fi
    done
done