# ProtAugment

## Running Experiments
To run the same experiments as in the paper, execute the following script:
```bash
chmod +x utils/scripts/protaugment/run_protaugment.sh
./utils/scripts/protaugment/run_protaugment.sh
```

Note that this script is made to be run on a cluster equipped with the [SLURM](https://slurm.schedmd.com/overview.html) software. 
If you don't use such software, remove the `sbatch <...>` commands prefixing the `models/proto/{protonet,protaugment}.sh` in the `run_protaugment.sh` script.