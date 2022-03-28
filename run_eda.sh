#!/bin/bash

#SBATCH -J eda                  # 作业名为 test
#SBATCH -o eda.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p inspur                # 分区参数
#SBATCH -w inspur-gpu-06               # 分区参数

for dataset in data/Reuters data/20News data/Amazon data/HuffPost; do
    for cv in 01 02 03 04 05; do
        python -u eda.py \
            --src-file ${dataset}/few_shot/${cv}/train.json \
            --tgt-file ${dataset}/few_shot/${cv}/train_aug.json
    done
done