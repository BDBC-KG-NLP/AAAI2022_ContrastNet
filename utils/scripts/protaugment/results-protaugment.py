#!.venv/bin/python
import os
import json
import numpy as np

ROOT_PATH = "runs_consistency/10samples"


def get_score_from_metrics_fp(metrics_fp):
    with open(os.path.join(metrics_fp), "r") as _f:
        metrics = json.load(_f)

    best_valid_acc = 0.0
    test_acc = 0.0
    assert len(metrics["valid"]) == len(metrics["test"])
    for valid_episode, test_episode in zip(metrics["valid"], metrics["test"]):
        for valid_metric_dict in valid_episode["metrics"]:
            if valid_metric_dict["tag"].startswith("acc"):
                valid_acc = valid_metric_dict["value"]
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    for test_metric_dict in test_episode["metrics"]:
                        if test_metric_dict["tag"].startswith("acc"):
                            test_acc = test_metric_dict["value"]
                        break
                    else:
                        raise ValueError
                break
        else:
            raise ValueError
    return test_acc


def find_results():
    out = list()
    for dataset in ("OOS", "Liu", "BANKING77", "HWU64"):
        # for C in (2, 5):
        for C in (5,):
            for K in (1, 5):
                # DBS
                for cv in ("01", "02", "03", "04", "05"):
                    for checkpoint_id in (6146,):
                        # for checkpoint_id in (6146, 12292):
                        dbs_root = f"runs_consistency/DBS-10samp/{dataset}/{cv}/{C}C_{K}K/seed42/paraphrase-checkpoint{checkpoint_id}"

                        if not os.path.exists(dbs_root):
                            continue
                        for d in os.listdir(dbs_root):
                            dbs_path = os.path.join(dbs_root, f"{d}/output")
                            if os.path.exists(os.path.join(dbs_path, "metrics.json")):
                                score = get_score_from_metrics_fp(os.path.join(dbs_path, "metrics.json"))
                                out.append({
                                    "score": score,
                                    "dataset": dataset,
                                    "C": C,
                                    "K": K,
                                    "method": f"{checkpoint_id}/DBS-{d}",
                                    "cv": cv,
                                    "split": "low"
                                })
                    back_translation_path = f"runs_consistency/DBS-10samp/{dataset}/{cv}/{C}C_{K}K/seed42/back-translation/output/metrics.json"
                    if os.path.exists(back_translation_path):
                        score = get_score_from_metrics_fp(back_translation_path)
                        out.append({
                            "score": score,
                            "dataset": dataset,
                            "C": C,
                            "K": K,
                            "method": f"back-translation",
                            "cv": cv,
                            "split": "low"
                        })

                    proto_path = f"runs_consistency/DBS-10samp/{dataset}/{cv}/{C}C_{K}K/seed42/proto-euclidean/output/metrics.json"
                    if os.path.exists(proto_path):
                        score = get_score_from_metrics_fp(proto_path)
                        out.append({
                            "score": score,
                            "dataset": dataset,
                            "C": C,
                            "K": K,
                            "method": f"proto",
                            "cv": cv,
                            "split": "low"
                        })

                    # full dataset
                    full_root = f"runs_consistency/full_datasets/{dataset}/{cv}/{C}C_{K}K/seed42"
                    for d in os.listdir(full_root):
                        run_path = os.path.join(full_root, f"{d}/output")
                        if os.path.exists(os.path.join(run_path, "metrics.json")):
                            score = get_score_from_metrics_fp(os.path.join(run_path, "metrics.json"))
                            out.append({
                                "score": score,
                                "dataset": dataset,
                                "C": C,
                                "K": K,
                                "method": d,
                                "cv": cv,
                                "split": "full"
                            })

    import pandas as pd

    df = pd.DataFrame(out)
    print(df.shape)
    df.to_clipboard(index=False)


if __name__ == "__main__":
    find_results()
