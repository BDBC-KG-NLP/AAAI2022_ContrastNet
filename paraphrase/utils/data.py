import numpy as np
import collections
from typing import List, Dict, Callable, Union
import logging
import torch

import random
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import BartTokenizerFast
from paraphrase.modeling import ParaphraseModel
from utils.data import get_json_data, get_txt_data
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
# from models.use import USEEmbedder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class FewShotDataset:
    def __init__(
            self,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            labels_path: str = None
    ):
        self.data_path = data_path
        self.labels_path = labels_path
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.data: Dict[str, List[Dict]] = None
        self.counter: Dict[str, int] = None
        self.load_file(data_path, labels_path)

    def load_file(self, data_path: str, labels_path: str = None):
        data = get_json_data(data_path)
        if labels_path:
            labels = get_txt_data(labels_path)
        else:
            labels = sorted(set([item["label"] for item in data]))

        labels_dict = collections.defaultdict(list)
        for item in data:
            if item["label"] in labels:
                labels_dict[item['label']].append(item)
        labels_dict = dict(labels_dict)

        for key, val in labels_dict.items():
            random.shuffle(val)
        self.data = labels_dict
        self.counter = {key: 0 for key, _ in self.data.items()}

    def get_episode(self) -> Dict:
        episode = dict()
        if self.n_classes:
            assert self.n_classes <= len(self.data.keys())
            rand_keys = np.random.choice(list(self.data.keys()), self.n_classes, replace=False)

            # Ensure enough data are query-able
            assert min([len(val) for val in self.data.values()]) >= self.n_support + self.n_query

            # Shuffle data
            for key in rand_keys:
                random.shuffle(self.data[key])

            if self.n_support:
                episode["xs"] = [[self.data[k][i] for i in range(self.n_support)] for k in rand_keys]
            if self.n_query:
                episode["xq"] = [[self.data[k][self.n_support + i] for i in range(self.n_query)] for k in rand_keys]
        return episode

    # def get_episode(self) -> Dict:
    #     episode = dict()
    #     if self.n_classes:
    #         assert self.n_classes <= len(self.data.keys())
    #         rand_keys = np.random.choice(list(self.data.keys()), self.n_classes, replace=False)
    #         task_rand_keys = [np.random.choice(list(self.data.keys()), self.n_classes, replace=False) for i in range(10)] + [rand_keys]

    #         # Ensure enough data are query-able
    #         assert min([len(val) for val in self.data.values()]) >= self.n_support + self.n_query

    #         # Shuffle data
    #         for key in rand_keys:
    #             random.shuffle(self.data[key])

    #         if self.n_support:
    #             episode["xs"] = [[self.data[k][i] for i in range(self.n_support)] for k in rand_keys]
    #             episode["task_xs"] = [[[self.data[k][i] for i in range(self.n_support)] for k in rand_key] for rand_key in task_rand_keys]
    #         if self.n_query:
    #             episode["xq"] = [[self.data[k][self.n_support + i] for i in range(self.n_query)] for k in rand_keys]


    #     return episode

    def __len__(self):
        return sum([len(label_data) for label, label_data in self.data.items()])


class FewShotPPDataset(FewShotDataset):
    def __init__(
            self,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            n_unlabeled: int,
            labels_path: str):
        super().__init__(data_path=data_path, n_classes=n_classes, n_support=n_support, n_query=n_query, labels_path=labels_path)
        self.n_unlabeled = n_unlabeled

    def get_episode(self) -> Dict:
        episode = super().get_episode()
        if self.n_classes:
            assert self.n_classes <= len(self.data.keys())
            rand_keys = np.random.choice(list(self.data.keys()), self.n_classes, replace=False)

            # Ensure enough data are query-able
            assert all(len(self.data[key]) >= self.n_support + self.n_query + self.n_unlabeled for key in rand_keys)

            # Shuffle data
            for key in rand_keys:
                random.shuffle(self.data[key])

            if self.n_support:
                episode["xs"] = [[self.data[k][i] for i in range(self.n_support)] for k in rand_keys]
            if self.n_query:
                episode["xq"] = [[self.data[k][self.n_support + i] for i in range(self.n_query)] for k in rand_keys]

            if self.n_unlabeled:
                episode['xu'] = [item for k in rand_keys for item in self.data[k][self.n_support + self.n_query:self.n_support + self.n_query + self.n_unlabeled]]

        return episode


class FewShotSSLFileDataset(FewShotDataset):
    def __init__(
            self,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            n_unlabeled: int,
            n_task: int,
            unlabeled_file_path: str,
            labels_path: str):
        super().__init__(data_path=data_path, n_classes=n_classes, n_support=n_support, n_query=n_query, labels_path=labels_path)
        self.n_unlabeled = n_unlabeled
        self.n_task = n_task
        logger.debug(f"Using augmented data @ {unlabeled_file_path}")
        self.unlabeled_data = get_json_data(unlabeled_file_path)
        logger.debug(f"Dataset has {len(self.unlabeled_data)} unlabeled samples")

    def get_episode(self) -> Dict:
        # Get episode from regular few-shot
        episode = super().get_episode()

        # task data
        task_rand_keys = [np.random.choice(list(self.data.keys()), self.n_classes, replace=False) for i in range(self.n_task)]
        episode["task_xs"] = [[[self.data[k][i] for i in range(1)] for k in rand_key] for rand_key in task_rand_keys] + [episode["xs"]]

        # Get random augmentations in the file
        unlabeled = np.random.choice(self.unlabeled_data, self.n_unlabeled).tolist()

        episode["x_augment"] = [
            {
                "src_text": u["src_text"],
                "tgt_texts": random.choice(u["tgt_texts"])
            }
            for u in unlabeled
        ]

        return episode


class FewShotSSLParaphraseDataset(FewShotDataset):
    n_unlabeled: int
    unlabeled_data: List[str]
    paraphrase_model: ParaphraseModel

    def __init__(
            self,
            data_path: str,
            n_classes: int,
            n_support: int,
            n_query: int,
            n_unlabeled: int,
            unlabeled_file_path: str,
            paraphrase_model: ParaphraseModel,
            labels_path: str):
        super().__init__(data_path=data_path, n_classes=n_classes, n_support=n_support, n_query=n_query, labels_path=labels_path)
        self.n_unlabeled = n_unlabeled
        self.unlabeled_data = get_txt_data(unlabeled_file_path)
        self.paraphrase_model = paraphrase_model

    def get_episode(self, **kwargs) -> Dict:
        episode = super().get_episode()

        # Get random augmentations in the file
        unlabeled = np.random.choice(self.unlabeled_data, self.n_unlabeled).tolist()
        tgt_texts = self.paraphrase_model.paraphrase(unlabeled, **kwargs)

        episode["x_augment"] = [
            {
                "src_text": src,
                "tgt_texts": tgts
            }
            for src, tgts in zip(unlabeled, tgt_texts)
        ]

        return episode
