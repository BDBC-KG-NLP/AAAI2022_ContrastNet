#!/usr/bin/env bash
cd $HOME/Projects/FewShotText
source .venv/bin/activate

for file in $(find data -name "*jsonl" | grep -P "(paraphrase|back)"); do
    filename=$(echo "$file" | grep -oP "[^/]+$")
    filename_no_ext=$(echo "$file" | grep -oP "[^/]+(?=.jsonl)")
    filedir=$(echo "$file" | grep -oP ".*(?=/$filename)")
    out_file="${filedir}/${filename_no_ext}-metrics.json"

    if [[ -f "${out_file}" ]]; then
        echo "${out_file} exists"
    else
        echo $filename $filename_no_ext $filedir $out_file
        PYTHONPATH=. python utils/scripts/paraphrase/evaluate-paraphrase-diversity.py \
            --in-file ${file} \
            --out-file ${out_file}

    fi

done