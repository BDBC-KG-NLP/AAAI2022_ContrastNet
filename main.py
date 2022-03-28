import logging

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

import json
import argparse

from transformers import AutoTokenizer

from model import ContrastNet

from paraphrase.utils.data import FewShotDataset, FewShotSSLFileDataset

from utils.data import get_json_data, FewShotDataLoader
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict, Callable, Union

from tensorboardX import SummaryWriter
import numpy as np
import warnings
from utils.few_shot import create_episode, create_ARSC_test_episode, create_ARSC_train_episode

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def run_fsintent(
        # Compulsory!
        data_path: str,
        train_labels_path: str,

        # Few-shot Stuff
        n_support: int,
        n_query: int,
        n_classes: int,
        model_name_or_path: str,
        super_tau: float = 1.0,
        unsuper_tau: float = 1.0,
        task_tau: float = 1.0,
        lr: float = 1e-6,
        metric: str = "euclidean",
        logger: object = None,
        super_weight: float = 0.7,
        task_weight: float = 1.0,
        max_len: int = 64,

        # Optional path to augmented data
        unlabeled_path: str = None,

        # Path training data ONLY (optional)
        train_path: str = None,

        # Validation & test
        valid_labels_path: str = None,
        test_labels_path: str = None,
        evaluate_every: int = 100,
        n_test_episodes: int = 1000,

        # Logging & Saving
        output_path: str = f'runs/{now()}',
        log_every: int = 10,

        # Training stuff
        max_iter: int = 10000,
        early_stop: int = None,

        # Encoder


        # Augmentation & paraphrase
        n_unlabeled: int = 5,
        n_task: int = 5,
        paraphrase_model_name_or_path: str = None,
        paraphrase_tokenizer_name_or_path: str = None,
        paraphrase_num_beams: int = None,
        paraphrase_beam_group_size: int = None,
        paraphrase_diversity_penalty: float = None,
        paraphrase_filtering_strategy: str = None,
        paraphrase_drop_strategy: str = None,
        paraphrase_drop_chance_speed: str = None,
        paraphrase_drop_chance_auc: float = None,

        paraphrase_generation_method: str = None,

        augmentation_data_path: str = None
):
    if output_path:
        if os.path.exists(output_path) and len(os.listdir(output_path)):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    # ----------
    # Load model
    # ----------

    fsinet: ContrastNet = ContrastNet(config_name_or_path=model_name_or_path, metric=metric, max_len=max_len, super_tau=super_tau, unsuper_tau=unsuper_tau, task_tau=task_tau)
    optimizer = torch.optim.Adam(fsinet.parameters(), lr=lr)

    logger.info(torch.cuda.list_gpu_processes())

    # ------------------
    # Load Train Dataset
    # ------------------
    train_dataset = FewShotSSLFileDataset(
        data_path=train_path if train_path else data_path,
        labels_path=train_labels_path,
        n_classes=n_classes,
        n_support=n_support,
        n_query=n_query,
        n_unlabeled=n_unlabeled,
        n_task=n_task,
        unlabeled_file_path=augmentation_data_path,
    )

    logger.info(f"Train dataset has {len(train_dataset)} items")

    # ---------
    # Load data
    # ---------
    logger.info(f"train labels: {train_dataset.data.keys()}")
    valid_dataset: FewShotDataset = None
    if valid_labels_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
        valid_dataset = FewShotDataset(data_path=data_path, labels_path=valid_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"valid labels: {valid_dataset.data.keys()}")
        assert len(set(valid_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    test_dataset: FewShotDataset = None
    if test_labels_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()
        test_dataset = FewShotDataset(data_path=data_path, labels_path=test_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"test labels: {test_dataset.data.keys()}")
        assert len(set(test_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0
    best_valid_dict = None
    best_test_dict = None

    for step in range(max_iter):

        # episode = train_dataset.get_episode()

        episode = train_dataset.get_episode()

        supervised_loss_share = super_weight*(1. - step/max_iter)
        task_loss_share = task_weight

        loss, loss_dict = fsinet.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share, task_loss_share=task_loss_share)

        fsinet.train_step(optimizer=optimizer, episode=episode, supervised_loss_share=supervised_loss_share, task_loss_share=task_loss_share)

        # for key, value in loss_dict["metrics"].items():
        #     if key!='com':
        #         train_metrics[key].append(value)

        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            for key, value in train_metrics.items():
                train_writer.add_scalar(tag=key, scalar_value=np.mean(value), global_step=step)
            logger.info(f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": key,
                        "value": np.mean(value)
                    }
                    for key, value in train_metrics.items()
                ],
                "global_step": step
            })

            train_metrics = collections.defaultdict(list)

        if valid_labels_path or test_labels_path:
            if (step + 1) % evaluate_every == 0:
                is_best = False
                for labels_path, writer, set_type, set_dataset in zip(
                        [valid_labels_path, test_labels_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_dataset, test_dataset]
                ):
                    if set_dataset:

                        set_results = fsinet.test_step(
                            dataset=set_dataset,
                            n_episodes=n_test_episodes
                        )

                        # set_set_results = fsinet.test_step(
                        #     dataset=set_dataset,
                        #     n_episodes=n_test_episodes
                        # )

                        # set_results = {}

                        # for key, value in set_set_results.items():
                        #     if key!='com':
                        #         set_results[key] = value
                                
                        # print(set_results)

                        # com = np.asarray(set_set_results['com'])

                        # # print(set_set_results['com'])
                        # print(np.sum(com)/(np.sum(com!=0)))

                        for key, val in set_results.items():
                            writer.add_scalar(tag=key, scalar_value=val, global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": key,
                                    "value": val
                                }
                                for key, val in set_results.items()
                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in set_results.items()]))
                        if set_type == "valid":
                            if set_results["acc"] >= best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                best_valid_dict = set_results
                                is_best = True
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")
                        else:
                            if is_best:
                                best_test_dict = set_results

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    logger.info(f"Best eval results: ")
                    logger.info(f"valid | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_valid_dict.items()]))
                    logger.info(f"test | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_test_dict.items()]))
                    break

    logger.info(f"Best eval results: ")
    logger.info(f"valid | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_valid_dict.items()]))
    logger.info(f"test | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_test_dict.items()]))

    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the full data")
    parser.add_argument("--train-labels-path", type=str, required=True, help="Path to train labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--train-path", type=str, help="Path to training data (if provided, picks training data from this path instead of --data-path")
    parser.add_argument("--model-name-or-path", type=str, default='bert-base-uncased', help="Language Model PROTAUGMENT initializes from")
    parser.add_argument("--lr", default=1e-6, type=float, help="Temperature of the contrastive loss")
    parser.add_argument("--super-tau", default=1.0, type=float, help="Temperature of the contrastive loss in supervised loss")
    parser.add_argument("--unsuper-tau", default=1.0, type=float, help="Temperature of the contrastive loss in instance-level regularizer")
    parser.add_argument("--task-tau", default=1.0, type=float, help="Temperature of the contrastive loss in task-level regularizer")
    parser.add_argument("--super-weight", default=0.7, type=float, help="The initialized supervised loss weight")
    parser.add_argument("--task-weight", default=1.0, type=float, help="The initialized supervised loss weight")
    parser.add_argument("--max-len", type=int, default=64, help="maxmium length of text sequence for BERT") 

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=1, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=1, help="Number of classes per episode")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance function to use", choices=("euclidean", "cosine"))
    parser.add_argument("--n-task", type=int, default=5, help="Number of tasks in task-level regularizer")

    # Validation & test
    parser.add_argument("--valid-labels-path", type=str, required=True, help="Path to valid labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--test-labels-path", type=str, required=True, help="Path to test labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--n-test-episodes", type=int, default=600, help="Number of episodes during evaluation (valid, test)")

    # Logging & Saving
    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--log-path", type=str, help="Path to the log file.")

    # Training stuff
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--early-stop", type=int, default=10, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Augmentation & Paraphrase
    parser.add_argument("--unlabeled-path", type=str, help="Path to raw data (one sentence per line), to generate paraphrases from.")
    parser.add_argument("--n-unlabeled", type=int, default=5, help="Number of rows to draw from `--unlabeled-path` at each episode")

    # If you are using a paraphrase generation model
    parser.add_argument("--paraphrase-model-name-or-path", default='tdopierre/ProtAugment-ParaphraseGenerator', type=str, help="Name or path to the paraphrase model")
    parser.add_argument("--paraphrase-tokenizer-name-or-path", default='tdopierre/ProtAugment-ParaphraseGenerator', type=str, help="Name or path to the paraphrase model's tokenizer")
    parser.add_argument("--paraphrase-num-beams", default=15, type=int, help="Total number of beams in the Beam Search algorithm")
    parser.add_argument("--paraphrase-beam-group-size", default=3, type=int, help="Size of each group of beams")
    parser.add_argument("--paraphrase-diversity-penalty", default=0.5, type=float, help="Diversity penalty (float) to use in Diverse Beam Search")
    parser.add_argument("--paraphrase-filtering-strategy", default='bleu', type=str, choices=["bleu", "clustering"], help="Filtering strategy to apply to a group of generated paraphrases to choose the one to pick. `bleu` takes the sentence which has the highest bleu_score w/r to the original sentence.")
    parser.add_argument("--paraphrase-drop-strategy", type=str, choices=["bigram", "unigram"], help="Drop strategy to use to contraint the paraphrase generation. If not set, no words are forbidden.")
    parser.add_argument("--paraphrase-drop-chance-speed", type=str, choices=["flat", "down", "up"], help="Curve of drop probability depending on token position in the sentence")
    parser.add_argument("--paraphrase-drop-chance-auc", type=float, help="Area of the drop chance probability w/r to the position in the sentence. When --paraphrase-drop-chance-speed=flat (same chance for all tokens to be forbidden no matter the position in the sentence), this parameter equals to p_{mask}")

    # If you want to use another augmentation technique, e.g. EDA (https://github.com/jasonwei20/eda_nlp/)
    parser.add_argument("--paraphrase-generation-method", type=str, choices=["eda"])

    # Augmentation file path (optional, but if provided it will be used)
    parser.add_argument("--augmentation-data-path", type=str, help="Path to a .json file containing augmentations. Refer to `back-translation.json` for an example")

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    warnings.simplefilter('ignore')

    handler = logging.FileHandler(args.log_path, mode='w')

    logger.addHandler(handler)

    logger.debug(f"Received args: {json.dumps(args.__dict__, sort_keys=True, ensure_ascii=False, indent=1)}")

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.data_path, args.train_labels_path, args.valid_labels_path, args.test_labels_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_fsintent(
        data_path=args.data_path,
        train_labels_path=args.train_labels_path,
        train_path=args.train_path,
        model_name_or_path=args.model_name_or_path,
        super_tau=args.super_tau,
        unsuper_tau=args.unsuper_tau,
        task_tau=args.task_tau,
        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        metric=args.metric,
        logger=logger,
        super_weight=args.super_weight,
        task_weight=args.task_weight,
        max_len=args.max_len,

        valid_labels_path=args.valid_labels_path,
        test_labels_path=args.test_labels_path,
        evaluate_every=args.evaluate_every,
        n_test_episodes=args.n_test_episodes,

        output_path=args.output_path,
        log_every=args.log_every,
        max_iter=args.max_iter,
        early_stop=args.early_stop,

        unlabeled_path=args.unlabeled_path,
        n_unlabeled=args.n_unlabeled,
        n_task=args.n_task,

        # Paraphrase generation model
        paraphrase_model_name_or_path=args.paraphrase_model_name_or_path,
        paraphrase_tokenizer_name_or_path=args.paraphrase_tokenizer_name_or_path,
        paraphrase_num_beams=args.paraphrase_num_beams,
        paraphrase_beam_group_size=args.paraphrase_beam_group_size,
        paraphrase_filtering_strategy=args.paraphrase_filtering_strategy,
        paraphrase_drop_strategy=args.paraphrase_drop_strategy,
        paraphrase_drop_chance_speed=args.paraphrase_drop_chance_speed,
        paraphrase_drop_chance_auc=args.paraphrase_drop_chance_auc,

        # Other paraphrase generation method
        paraphrase_generation_method=args.paraphrase_generation_method,

        # Or just path to augmented data
        augmentation_data_path=args.augmentation_data_path
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
