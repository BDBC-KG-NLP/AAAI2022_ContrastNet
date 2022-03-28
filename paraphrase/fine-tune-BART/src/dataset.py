import numpy as np
import collections
from typing import List, Dict, Callable, Union

import random
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import BartTokenizerFast

from .utils import get_jsonl_data, get_txt_data
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
import time
from .use import use_embedder
from paraphrase.modeling import DropChances

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ParaphraseModel:
    def paraphrase(self, **kwargs):
        raise NotImplementedError


class BaseParaphraseModel(ParaphraseModel):
    def __init__(
            self,
            model_name_or_path: str,
            tok_name_or_path: str = None,
            num_return_sequences: int = 1,
            num_beams: int = None
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
        self.tok = AutoTokenizer.from_pretrained(tok_name_or_path if tok_name_or_path else model_name_or_path)
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams if num_beams else self.num_return_sequences
        assert self.num_beams >= self.num_return_sequences

    def paraphrase(self, src_texts):
        batch = self.tok.prepare_seq2seq_batch(
            src_texts=src_texts,
            max_length=512,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = self.model.generate(**batch, max_length=512, num_beams=self.num_beams, num_return_sequences=self.num_return_sequences)
        tgt_texts = self.tok.batch_decode(preds.detach().cpu(), skip_special_tokens=True)
        return [tgt_texts[i:i + self.num_return_sequences] for i in range(0, len(src_texts) * self.num_return_sequences, self.num_return_sequences)]



class ForbidStrategies:
    def __init__(self, special_ids: List[int]):
        self.special_ids = special_ids

    def unigram_dropping_strategy(self, input_ids: torch.Tensor, drop_chance_fn: Callable):
        bad_words_ids = list()
        for row in input_ids.tolist():
            row = [item for item in row if item not in self.special_ids]
            for item_ix, item in enumerate(row):
                drop_chance = drop_chance_fn(item_ix, len(row))
                if random.random() < drop_chance:
                    bad_words_ids.append(item)

        # Reshape to correct format
        bad_words_ids = [[item] for item in bad_words_ids]
        return bad_words_ids

    def bigram_dropping_strategy(self, input_ids: torch.Tensor):
        bad_words_ids = list()
        for row in input_ids.tolist():
            row = [item for item in row if item not in self.special_ids]
            for i in range(0, len(row) - 1):
                bad_words_ids.append(row[i:i + 2])
        return bad_words_ids


class BaseParaphraseBatchPreparer:
    def __init__(self, tokenizer: BartTokenizerFast):
        self.tokenizer = tokenizer

    def prepare_batch(self, src_texts: List[str]):
        batch = self.tokenizer.prepare_seq2seq_batch(src_texts=src_texts, return_tensors="pt", max_length=512)
        batch = {k: v.to(device) for k, v in batch.items()}
        self.pimp_batch(batch)
        return batch

    def pimp_batch(self, batch: Dict[str, torch.Tensor], **kwargs):
        # This must be implemented elsewhere!
        return


class UnigramRandomDropParaphraseBatchPreparer(BaseParaphraseBatchPreparer):

    def __init__(self, tokenizer: BartTokenizerFast, auc: float = None, drop_chance_speed: str = None):
        super().__init__(tokenizer=tokenizer)

        # Args checking
        self.auc = auc
        assert 0 <= self.auc <= 1
        self.drop_chance_speed = drop_chance_speed
        assert self.drop_chance_speed in ("flat", "slow", "fast", "up")

    def pimp_batch(self, batch: Dict[str, torch.Tensor], **kwargs):
        bad_words_ids = ForbidStrategies(
            special_ids=self.tokenizer.all_special_ids
        ).unigram_dropping_strategy(
            batch["input_ids"],
            drop_chance_fn=DropChances(auc=self.auc).get_drop_fn(self.drop_chance_speed)
        )
        if len(bad_words_ids):
            batch["bad_words_ids"] = bad_words_ids


class BigramDropParaphraseBatchPreparer(BaseParaphraseBatchPreparer):
    def __init__(self, tokenizer: BartTokenizerFast):
        super().__init__(tokenizer=tokenizer)

    def pimp_batch(self, batch: Dict[str, torch.Tensor], **kwargs):
        bad_words_ids = ForbidStrategies(special_ids=self.tokenizer.all_special_ids).bigram_dropping_strategy(batch["input_ids"])
        if len(bad_words_ids):
            batch["bad_words_ids"] = bad_words_ids


def tune_batch_random_drop(batch: Dict[str, torch.Tensor], drop_prob: float = 1):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    ids_filtered = input_ids * attention_mask * (input_ids != 2)
    ids_filtered = (torch.rand_like(ids_filtered.to(float)) > 1 - drop_prob) * ids_filtered
    ids_filtered = ids_filtered[ids_filtered != 0]
    bad_words_ids = ids_filtered.view(-1).unique().view(-1, 1).tolist()
    if len(bad_words_ids):
        batch["bad_words_ids"] = bad_words_ids


def bleu_score(src: str, dst: str):
    from sacrebleu import sentence_bleu
    return sentence_bleu(dst, [src]).score


def filter_generated_texts_with_clustering(texts: List[str], n_return_sequences: int):
    assert len(texts) >= n_return_sequences
    embeddings = use_embedder.embed_many(texts)

    # KMeans (this is too slow)
    # from sklearn.cluster import KMeans, AgglomerativeClustering
    # clustering_algo = KMeans(n_clusters=self.num_beam_groups, max_iter=10)

    # Agglomerative Clustering
    from sklearn.cluster import AgglomerativeClustering
    clustering_algo = AgglomerativeClustering(n_clusters=n_return_sequences, affinity='euclidean', linkage='ward')
    labels = clustering_algo.fit_predict(embeddings)

    # Organise labels & data into clusters
    cluster = dict()
    for txt_ix, (txt, label) in enumerate(zip(texts, labels)):
        cluster.setdefault(label, []).append((txt_ix, txt))

    # Write to output
    output = list()
    from sklearn.metrics.pairwise import pairwise_distances
    for label, txts in cluster.items():
        # In a cluster, select the sentence closest to the center
        distances = pairwise_distances([
            embeddings[txt_ix] for txt_ix, _ in txts
        ], [embeddings[labels == label].mean(0)])
        output.append(txts[distances.flatten().argmin()][1])
        # batch_output.append(txts[0])
    return output


def filter_generated_texts_with_distance_metric(texts: List[List[str]], src: str, distance_metric_fn: Callable[[str, str], float], lower_is_better: bool = True):
    scores = [
        [distance_metric_fn(src, text) for text in group]
        for group in texts
    ]

    if lower_is_better:
        ranking_fn = np.argmin
    else:
        ranking_fn = np.argmax
    return [
        group[ranking_fn(scores_)]
        for group, scores_ in zip(texts, scores)
    ]


class DBSParaphraseModel(ParaphraseModel):
    def __init__(
            self,
            model_name_or_path: str,
            tok_name_or_path: str = None,
            beam_group_size: int = 4,
            num_beams: int = 20,
            filtering_strategy: str = None,
            paraphrase_batch_preparer: BaseParaphraseBatchPreparer = None
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name_or_path if tok_name_or_path else model_name_or_path)
        self.num_return_sequences = self.num_beams = num_beams
        self.beam_group_size = beam_group_size
        self.num_beam_groups = self.num_beams // self.beam_group_size
        assert self.num_beams % self.beam_group_size == 0
        self.filtering_strategy = filtering_strategy
        if paraphrase_batch_preparer is None:
            paraphrase_batch_preparer = BaseParaphraseBatchPreparer(tokenizer=self.tokenizer)
        self.paraphrase_batch_preparer = paraphrase_batch_preparer

    def paraphrase(self, src_texts):
        batch = self.paraphrase_batch_preparer.prepare_batch(src_texts=src_texts)
        preds = self.model.generate(**batch, max_length=512, num_beams=self.num_beams, num_beam_groups=self.beam_group_size, num_return_sequences=self.num_return_sequences)
        tgt_texts = self.tokenizer.batch_decode(preds.detach().cpu(), skip_special_tokens=True)

        batches = [tgt_texts[i:i + self.num_return_sequences] for i in range(0, len(src_texts) * self.num_return_sequences, self.num_return_sequences)]

        output = list()

        for src, batch in zip(src_texts, batches):
            if self.filtering_strategy == "clustering":
                filtered = filter_generated_texts_with_clustering(batch, self.num_beam_groups)
            elif self.filtering_strategy == "bleu":
                filtered = filter_generated_texts_with_distance_metric(
                    texts=[batch[i:i + self.beam_group_size] for i in range(0, len(batch), self.beam_group_size)],
                    src=src,
                    distance_metric_fn=bleu_score,
                    lower_is_better=True
                )
            else:
                raise ValueError
            output.append(filtered)

        return output
