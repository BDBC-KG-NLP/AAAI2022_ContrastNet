import pandas as pd
from utils.data import get_jsonl_data
import os
import random
import numpy as np

random.seed(42)
np.random.seed(42)

writer = pd.ExcelWriter("paraphrase_samples.xlsx", engine='xlsxwriter')


for dataset in ("OOS", "Liu", "BANKING77","HWU64", ):
    dataset_path = f"data/{dataset}"
    sentences_from_bt = get_jsonl_data(os.path.join(dataset_path, "back-translations.jsonl"))
    data = list()
    for d in sentences_from_bt:
        for t in d["tgt_texts"]:
            data.append({
                "src": d["src_text"],
                "tgt": t,
                "method": "bt"
            })
    for root, folders, files in os.walk(dataset_path):
        for file in files:
            if file == "paraphrases.jsonl":
                run_name = os.path.split(root)[-1]
                for d in get_jsonl_data(os.path.join(root, file)):

                    for t in d["tgt_texts"]:
                        data.append({
                            "src": d["src_text"],
                            "tgt": t,
                            "method": run_name
                        })
    df = pd.DataFrame(data)
    df_filtered = df[df.src.isin(np.random.choice(df.src.unique(), size=1000))]
    df_filtered.sort_values(by=["src", "method"]).to_excel(writer, index=False, sheet_name=dataset)
writer.close()