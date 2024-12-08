#!/bin/bash
#SBATCH -p general
#SBATCH -t 2-00:00:00
#SBATCH -G a100:1
#SBATCH -C a100_80
#SBATCH --mail-type=ALL
#SBATCH --mail-user=phegde7@asu.edu
#SBATCH -o logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate /scratch/phegde7/.conda/envs/vq

python3 unlearn_harm.py --model_name=state-spaces/mamba-1.4b-hf --model_save_dir=models/mamba-1.4b_unlearned_ga_20000 --log_file=logs/mamba-1.4b-unlearn_ga_20000.log --save_every=500 --batch_size=1 --hf_cache_dir=/scratch/phegde7/.cache/huggingface/ --max_unlearn_steps=20000 --max_bad_loss=100000 --random_weight=0 --normal_weight=0
