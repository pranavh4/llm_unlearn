#!/bin/bash
#SBATCH -p general
#SBATCH -t 2-00:00:00
#SBATCH -G 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=phegde7@asu.edu
#SBATCH -o logs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e logs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate /scratch/phegde7/.conda/envs/vq


python3 evaluate.py --model_name=state-spaces/mamba-1.4b-hf --model_path=/scratch/phegde7/CSE_575/llm_unlearn/models/mamba-1.4b_unlearned_ga_20000/checkpoint_20000 --output_file_prefix=mamba-1.4b_unlearned_ga_20000 --num_samples=200

python3 evaluate.py --model_name=state-spaces/mamba-1.4b-hf --model_path=/scratch/phegde7/CSE_575/llm_unlearn/models/mamba-1.4b_unlearned_ga_20000/checkpoint_2000 --output_file_prefix=mamba-1.4b_unlearned_ga_2000 --num_samples=200


python3 evaluate.py --model_name=state-spaces/mamba-1.4b-hf --model_path=/scratch/phegde7/CSE_575/llm_unlearn/models/mamba-1.4b_unlearned_ga_20000/checkpoint_500 --output_file_prefix=mamba-1.4b_unlearned_ga_500 --num_samples=200
