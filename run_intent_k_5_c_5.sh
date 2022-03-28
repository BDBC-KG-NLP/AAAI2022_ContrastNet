#!/bin/bash

#SBATCH -J Instance                  # 作业名为 test
#SBATCH -o Instance.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p inspur                # 分区参数
#SBATCH -w inspur-gpu-05               # 分区参数



dataset=data/BANKING77
tuned=tdopierre/ProtAugment-LM-BANKING77
augmention=data/BANKING77/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_K_5_C_5/BANKING77/K_5_C_5
logfile=K_5_C_5
K=5
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for cv in 01 02 03 04 05; do
    python -u main.py \
        --data-path ${dataset}/full.json \
        --train-path ${dataset}/few_shot/${cv}/train_aug.json \
        --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
        --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
        --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
        --unlabeled-path ${dataset}/raw.txt \
        --log-path ${logdir}/${cv}.out \
        \
        --n-support ${K} \
        --n-query ${K} \
        --n-classes ${C} \
        \
        --super-tau 5.0 \
        --unsuper-tau 7.0 \
        --task-tau 7.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 600 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 20 \
        --seed 42 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 10 \
        --n-task 10 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done

dataset=data/HWU64
tuned=tdopierre/ProtAugment-LM-HWU64
augmention=data/HWU64/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_K_5_C_5/HWU64/K_5_C_5
logfile=K_5_C_5
K=5
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for cv in 01 02 03 04 05; do
    python -u main.py \
        --data-path ${dataset}/full.json \
        --train-path ${dataset}/few_shot/${cv}/train_aug.json \
        --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
        --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
        --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
        --unlabeled-path ${dataset}/raw.txt \
        --log-path ${logdir}/${cv}.out \
        \
        --n-support ${K} \
        --n-query ${K} \
        --n-classes ${C} \
        \
        --super-tau 5.0 \
        --unsuper-tau 7.0 \
        --task-tau 7.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 600 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 20 \
        --seed 42 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 10 \
        --n-task 10 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done

dataset=data/Liu
tuned=tdopierre/ProtAugment-LM-Liu
augmention=data/Liu/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_K_5_C_5/Liu/K_5_C_5
logfile=K_5_C_5
K=5
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for cv in 01 02 03 04 05; do
    python -u main.py \
        --data-path ${dataset}/full.json \
        --train-path ${dataset}/few_shot/${cv}/train_aug.json \
        --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
        --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
        --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
        --unlabeled-path ${dataset}/raw.txt \
        --log-path ${logdir}/${cv}.out \
        \
        --n-support ${K} \
        --n-query ${K} \
        --n-classes ${C} \
        \
        --super-tau 5.0 \
        --unsuper-tau 7.0 \
        --task-tau 7.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 600 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 20 \
        --seed 42 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 10 \
        --n-task 10 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done

dataset=data/OOS
tuned=tdopierre/ProtAugment-LM-Clinic150
augmention=data/OOS/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_K_5_C_5/OOS/K_5_C_5
logfile=K_5_C_5
K=5
C=5

if [ ! -d ${logdir}  ];then mkdir -p ${logdir};fi

for cv in 01 02 03 04 05; do
    python -u main.py \
        --data-path ${dataset}/full.json \
        --train-path ${dataset}/few_shot/${cv}/train_aug.json \
        --train-labels-path ${dataset}/few_shot/${cv}/labels.train.txt \
        --valid-labels-path ${dataset}/few_shot/${cv}/labels.valid.txt \
        --test-labels-path ${dataset}/few_shot/${cv}/labels.test.txt \
        --unlabeled-path ${dataset}/raw.txt \
        --log-path ${logdir}/${cv}.out \
        \
        --n-support ${K} \
        --n-query ${K} \
        --n-classes ${C} \
        \
        --super-tau 5.0 \
        --unsuper-tau 7.0 \
        --task-tau 7.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 600 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 20 \
        --seed 42 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 10 \
        --n-task 10 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done
