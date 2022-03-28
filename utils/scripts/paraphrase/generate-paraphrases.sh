#!/usr/bin/env bash
#SBATCH --mail-user=thomas.dopierre@hotmail.fr
#SBATCH --mail-type=FAIL
#SBATCH -t 1-00:00:00
#SBATCH --mem=20G
cd $HOME/Projects/FewShotText
source .venv/bin/activate
source .envrc
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "hostname: $(hostname)"
command -v nvidia-smi >/dev/null && {
    echo "GPU Devices:"
    nvidia-smi
} || {
    :
}

PYTHONPATH=. python utils/scripts/paraphrase/generate-paraphrases.py $@