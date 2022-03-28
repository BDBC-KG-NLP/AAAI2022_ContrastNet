import json
from typing import List

from sacrebleu import corpus_bleu
import argparse

from tqdm import tqdm

from models.use import use_embedder
import sacrebleu
from utils.data import get_jsonl_data
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


def dist_k(texts: List[str], k: int, lowercase: bool = False) -> float:
    if lowercase:
        texts = [t.lower() for t in texts]
    splitted = [
        t.strip().split()
        for t in texts
    ]
    k_grams = [
        tuple(s[i:i + k])
        for s in splitted for i in range(0, len(s) - k + 1) if len(s) >= k
    ]
    n_distinct_k_grams = len(set(k_grams))
    n_tokens = sum([len(s) for s in splitted])
    return n_distinct_k_grams / n_tokens


def main():
    args = parse_args()
    data = get_jsonl_data(args.in_file)

    metrics = dict()

    # Group sentence into sys and refs
    sys = [d["src_text"] for d in data]
    refs = [[] for _ in range(5)]
    for d in data:
        for ix, tgt_text in enumerate(d["tgt_texts"]):
            refs[ix].append(tgt_text)

    # Compute bleu score
    bleus = [sacrebleu.corpus_bleu(sys, [refs[i]]).score for i in tqdm(range(5))]
    metrics["bleus"] = bleus
    metrics["bleu"] = np.mean(bleus)

    # Sys embed
    sys_embed = use_embedder.embed_many(sys)
    tgt_embed = use_embedder.embed_many([t for d in data for t in d["tgt_texts"]]).reshape(sys_embed.shape[0], 5, -1)

    # Compute USE similarities
    use_similarities = list()
    for s_embed, t_embed in tqdm(zip(sys_embed, tgt_embed)):
        m = np.concatenate((s_embed.reshape(1, -1), t_embed))
        dists = pairwise_distances(m, metric="cosine")

        use_similarities.append((1 - dists).sum() / 36)

    # Compute dist-k
    for k in (2, 3):
        for lower in (False, True):
            dists = list()
            for r in tqdm(refs):
                dist_k_ = dist_k(r, k=k, lowercase=lower)
                dists.append(dist_k_)
            dists = np.mean(dists)
            metrics[f"dist-{k}{'_lowercased' if lower else ''}"] = dists

            dists = list()
            for ix_r, r in enumerate(tqdm(refs)):
                dist_k_with_s = dist_k(r + [sys[ix_r]], k=k, lowercase=lower)
                dists.append(dist_k_with_s)
            dists = np.mean(dists)
            metrics[f"dist-{k}_with_orig{'_lowercased' if lower else ''}"] = dists

    metrics["use_similarity"] = np.mean(use_similarities)
    import os
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as file:
        json.dump(metrics, file, indent=1, ensure_ascii=False)
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
