#!/bin/bash

#SBATCH -J Instance                  # 作业名为 test
#SBATCH -o Instance.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p inspur                # 分区参数
#SBATCH -w inspur-gpu-05               # 分区参数



dataset=data/HuffPost
tuned=bert-base-uncased
augmention=data/HuffPost/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_instance/HuffPost/K_1_C_5
logfile=K_1_C_5
K=1
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
        --task-tau 3.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 1000 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 10 \
        --seed 42 \
        --max-len 256 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 2 \
        --n-task 2 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done


dataset=data/Reuters
tuned=bert-base-uncased
augmention=data/Reuters/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_instance/Reuters/K_1_C_5
logfile=K_1_C_5
K=1
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
        --task-tau 3.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 1000 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 10 \
        --seed 42 \
        --max-len 256 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 2 \
        --n-task 2 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done



dataset=data/20News
tuned=bert-base-uncased
augmention=data/20News/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_instance/20News/K_1_C_5
logfile=K_1_C_5
K=1
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
        --task-tau 3.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 1000 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 10 \
        --seed 42 \
        --max-len 256 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 2 \
        --n-task 2 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done



dataset=data/Amazon
tuned=bert-base-uncased
augmention=data/Amazon/paraphrases/DBS-unigram-flat-1.0/paraphrases.json


logdir=log_instance/Amazon/K_1_C_5
logfile=K_1_C_5
K=1
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
        --task-tau 3.0 \
        --lr 1e-6 \
        --super-weight 0.95 \
        --task-weight 0.1 \
        --evaluate-every 100 \
        --n-test-episodes 1000 \
        --log-every 10 \
        --max-iter 10000 \
        --early-stop 10 \
        --seed 42 \
        --max-len 256 \
        \
        --model-name-or-path ${tuned} \
        --n-unlabeled 2 \
        --n-task 2 \
        --augmentation-data-path ${augmention} \
        \
        --metric euclidean \
        --supervised-loss-share-power 1
done