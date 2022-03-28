import os
from paraphrase.modeling import UnigramRandomDropParaphraseBatchPreparer, BigramDropParaphraseBatchPreparer, BaseParaphraseBatchPreparer, DBSParaphraseModel
from transformers import AutoTokenizer
import torch
import argparse
import logging
from tqdm import tqdm

from utils.data import write_jsonl_data

logging.basicConfig()
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', required=True, type=str)
    parser.add_argument('--tgt-file', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=16)

    # Augmentation & Paraphrase
    parser.add_argument("--paraphrase-model-name-or-path", type=str)
    parser.add_argument("--paraphrase-tokenizer-name-or-path", type=str)
    parser.add_argument("--paraphrase-num-beams", type=int)
    parser.add_argument("--paraphrase-beam-group-size", type=int)
    parser.add_argument("--paraphrase-diversity-penalty", type=float)
    parser.add_argument("--paraphrase-filtering-strategy", type=str)
    parser.add_argument("--paraphrase-drop-strategy", type=str)
    parser.add_argument("--paraphrase-drop-chance-speed", type=str)
    parser.add_argument("--paraphrase-drop-chance-auc", type=float)

    args = parser.parse_args()
    assert os.path.exists(args.src_file)
    assert not os.path.exists(args.tgt_file)
    return args


def main():
    # Load arguments
    args = parse_args()

    # Set physical device which will be used
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}")

    # Load paraphrase model
    paraphrase_tokenizer = AutoTokenizer.from_pretrained(args.paraphrase_tokenizer_name_or_path)
    if args.paraphrase_drop_strategy == "unigram":
        paraphrase_batch_preparer = UnigramRandomDropParaphraseBatchPreparer(
            tokenizer=paraphrase_tokenizer,
            auc=args.paraphrase_drop_chance_auc,
            drop_chance_speed=args.paraphrase_drop_chance_speed,
            device=device
        )
    elif args.paraphrase_drop_strategy == "bigram":
        paraphrase_batch_preparer = BigramDropParaphraseBatchPreparer(tokenizer=paraphrase_tokenizer, device=device)
    else:
        paraphrase_batch_preparer = BaseParaphraseBatchPreparer(tokenizer=paraphrase_tokenizer, device=device)

    paraphrase_model = DBSParaphraseModel(
        model_name_or_path=args.paraphrase_model_name_or_path,
        tok_name_or_path=args.paraphrase_tokenizer_name_or_path,
        num_beams=args.paraphrase_num_beams,
        beam_group_size=args.paraphrase_beam_group_size,
        diversity_penalty=args.paraphrase_diversity_penalty,
        filtering_strategy=args.paraphrase_filtering_strategy,
        paraphrase_batch_preparer=paraphrase_batch_preparer,
        device=device
    )

    # Open input file
    with open(args.src_file, "r") as file:
        lines_in = [line.strip() for line in file]
    out = list()

    for i in tqdm(range(0, len(lines_in), args.batch_size)):
        lines_batch = lines_in[i:i + args.batch_size]
        paraphrases = paraphrase_model.paraphrase(src_texts=lines_batch)
        for line, p in zip(lines_batch, paraphrases):
            out.append({
                "src_text": line,
                "tgt_texts": p
            })

    os.makedirs(os.path.dirname(args.tgt_file), exist_ok=True)
    write_jsonl_data(out, args.tgt_file)

    #
    # # Load model, tokenizer
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(torch.device("cuda"))
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    #
    # # Load dataset
    # dataset = PredictionSeq2SeqDataset(
    #     tokenizer=tokenizer,
    #     src_path=args.src_file,
    #     max_source_length=args.max_source_length,
    #     max_target_length=args.max_target_length
    # )
    #
    # # Generate
    # output = list()
    #
    # for i in tqdm(range(0, len(dataset), args.batch_size)):
    #     dataset_batch = [dataset[ix] for ix in range(i, min(len(dataset), i + args.batch_size))]
    #     batch = dataset.collate_fn(dataset_batch)
    #     for k, v in batch.items():
    #         batch[k] = v.to(torch.device("cuda"))
    #     batch.pop('ids')
    #     preds = model.generate(**batch, num_return_sequences=args.num_return_sequences, num_beams=args.num_beams)
    #     decodes = tokenizer.batch_decode(preds.detach().cpu(), skip_special_tokens=True)
    #     for j, d in enumerate(dataset_batch):
    #         output.append({
    #             "src": d,
    #             "tgt": decodes[args.num_return_sequences * j: args.num_return_sequences * (j + 1)]
    #         })
    #
    # logger.info(f'Writing {len(output)} items to {args.tgt_file}')
    # write_jsonl_data(output, args.tgt_file)


if __name__ == "__main__":
    main()
