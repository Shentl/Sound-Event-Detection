#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module load miniconda3
source activate
conda activate common

python -u run.py train_evaluate configs/baseline_all_para.yaml data/eval/feature.csv data/eval/label.csv
