import json
import os

out = list()
for dataset in ("BANKING77", "HWU64", "OOS", "Liu"):
    # Back-translation metrics
    with open(f"data/{dataset}/back-translations-metrics.json", "r") as file:
        metrics = json.load(file)
        metrics["method"] = "back-translation"
        metrics["dataset"] = dataset
        out.append(metrics)

    for method in os.listdir(f"data/{dataset}/paraphrases"):
        with open(f"data/{dataset}/paraphrases/{method}/paraphrases-metrics.json", "r") as file:
            metrics = json.load(file)
            metrics["method"] = method
            metrics["dataset"] = dataset
            out.append(metrics)

import pandas as pd

df = pd.DataFrame(out)
print(df.shape)
df.to_clipboard(index=False)
