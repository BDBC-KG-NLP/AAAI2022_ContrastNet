import tqdm
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_file(path):
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


def write_file(lines, path, exist_ok=False):
    if os.path.exists(path) and not exist_ok:
        raise FileExistsError
    with open(path, "w") as file:
        for line in lines:
            file.write(f"{line}\n")


class Translator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(get_device())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def translate(self, text):
        return self.translate_multiple([text])[0]

    def translate_multiple(self, texts):
        inputs = self.tokenizer(texts, padding="longest", return_tensors="pt", truncation=True, max_length=self.model.config.max_length)['input_ids'].to(get_device())
        with torch.no_grad():
            outputs = self.model.generate(inputs)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


class BackTranslator:
    def __init__(self, model_name_fw: str = None, model_name_bw: str = None):
        self.fw_model = Translator(model_name=model_name_fw)
        self.bw_model = Translator(model_name=model_name_bw)

    def process(self, text):
        return self.process_multiple([text])[0]

    def process_multiple(self, texts):
        fw = self.fw_model.translate_multiple(texts)
        bw = self.bw_model.translate_multiple(fw)
        return bw


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-in", type=str, required=True)
    parser.add_argument("--file-out", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    # Checking args
    if not os.path.exists(args.file_in):
        raise FileNotFoundError(args.file_in)

    if os.path.exists(args.file_out) and not args.overwrite:
        raise FileExistsError(args.file_out)

    lines_in = read_file(args.file_in)
    translator = Translator(model_name=args.model_name)
    n_lines_in = len(lines_in)
    batch_size = 32
    lines_out = list()

    progress_bar = tqdm.tqdm(total=n_lines_in)

    for i in range(0, n_lines_in, batch_size):
        lines_batch = lines_in[i:i + batch_size]
        lines_out += translator.translate_multiple(texts=lines_batch)
        progress_bar.update(len(lines_batch))

    progress_bar.close()
    write_file(lines=lines_out, path=args.file_out)


if __name__ == "__main__":
    main()


def test_model():
    tr = Translator("Helsinki-NLP/opus-mt-fr-en")
    texts = ["Salut, ca va bien ?"] * 10 + ["oui, ça va parfaitement bien !! et toi?"] * 10
    print(tr.translate_multiple(texts))

    inputs = tr.tokenizer(texts, padding="longest", return_tensors="pt")['input_ids'].to(torch.device("cuda"))
    outputs = tr.model.generate(inputs, num_beams=4, max_length=40, early_stopping=True)

    tr.tokenizer.decode(outputs[0,], skip_special_tokens=True)

    back_translator = BackTranslator(
        model_name_fw="Helsinki-NLP/opus-mt-fr-en",
        model_name_bw="Helsinki-NLP/opus-mt-en-fr"
    )

    back_translator.process()
