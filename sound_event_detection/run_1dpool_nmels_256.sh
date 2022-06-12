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

python -u run.py train_evaluate configs/1dpool_nmels_256.yaml data_new/eval/feature.csv data_new/eval/label.csv
