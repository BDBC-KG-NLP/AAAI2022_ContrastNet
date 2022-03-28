# Fine-tuning BART on the paraphrasing task

Here is the code to fine-tune a **BART** pre-trained model on the paraphrasing task. 

## Datasets 
To train BART on the paraphrasing task so, we use 3 datasets:
- Quora
- MSR
- Google PAWS-Wiki

We build a dataset using a mix of the three. To mitigate the impact of Quora (it is huge), we truncate it so that it represents at most 50% of the final dataset.
The data is available in the directory `data/paraphrase/balanced`

## Training
To train the model, execute the following script
```bash
chmod +x bin/train-paraphrase-bart.sh
./bin/train-paraphrase-bart.sh
```

The paraphrase model will be put in the `runs/paraphrase/balanced/` folder.

If you have access to a machine capable of receiving [SLURM](https://slurm.schedmd.com/overview.html) commands, you can use the following script:

```bash
chmod +x bin/train-paraphrase-bart-slurm.sh
./bin/train-paraphrase-bart-slurm.sh
```


If you want to tweak the training parameters, you can directly used the `bin/train.sh` script, here is an example :arrow_heading_down:
```bash
./bin/train.sh \
    --model_name_or_path "facebook/bart-base" \
    --output_dir <output_dir> \
    --run_name <run_name> \
    \
    --do_train \
    --data_dir <dataset_path> \
    --num_train_epochs 20 \
    --warmup_steps 10000 \
    --logging_dir <logging_dir> \
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
    --seed 42
```



