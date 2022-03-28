#!/usr/bin/env bash
cd $HOME/Projects/FewShotText
source .venv/bin/activate
source .envrc

cd paraphrase/fine-tune-BART
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
command -v nvidia-smi >/dev/null && {
    echo "GPU Devices:"
    nvidia-smi
} || {
    :
}

PYTHONPATH=. python main.py $@
