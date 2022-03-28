import json
import argparse


def read_txt(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


def read_jsonl(path):
    with open(path, "r") as file:
        return [json.loads(line.strip()) for line in file.readlines()]


def write_jsonl(data, path):
    with open(path, "w") as file:
        for d in data:
            file.write(json.dumps(d, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", type=str, required=True)
    parser.add_argument("--file-out", type=str, required=True)
    parser.add_argument("-a", "--augmentation-path", action="append", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = read_txt(args.file_in)
    out = [
        {
            "src_text": src,
            "tgt_texts": []
        } for src in source
    ]

    for augmentation_path in args.augmentation_path:
        augmentations = read_txt(augmentation_path)
        assert len(augmentations) == len(source)

        for o, a in zip(out, augmentations):
            o["tgt_texts"].append(a)

    write_jsonl(out, args.file_out)


if __name__ == "__main__":
    main()
