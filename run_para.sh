#!/bin/bash

#SBATCH -J para                  # 作业名为 test
#SBATCH -o para.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p inspur                # 分区参数
#SBATCH -w inspur-gpu-06               # 分区参数



dataset=data/BANKING77
tuned=tdopierre/ProtAugment-ParaphraseGenerator
augmention=data/BANKING77/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_para
logfile=K_1_C_5
K=1
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for dataset in data/Reuters data/20News data/Amazon; do
    for cv in 01 02 03 04 05; do
        python -u generate-paraphrases.py \
            --src-file ${dataset}/few_shot/${cv}/train.json \
            --tgt-file ${dataset}/few_shot/${cv}/train_aug.json \
            --batch-size 8 \
            --paraphrase-model-name-or-path ${tuned} \
            --paraphrase-tokenizer-name-or-path ${tuned} \
            --paraphrase-num-beams 15 \
            --paraphrase-beam-group-size 3 \
            --paraphrase-diversity-penalty 0.5 \
            --paraphrase-filtering-strategy bleu \
            --paraphrase-drop-strategy unigram \
            --paraphrase-drop-chance-speed flat \
            --paraphrase-drop-chance-auc 1.0
    done
done