#!/bin/bash

#SBATCH -p 64c512g
#SBATCH -N 1
#SBATCH -n 1

module load miniconda3
source activate
conda activate common

mkdir {dev,eval,metadata}

# development set
echo "Preparing development set"
python prepare_wav_csv.py "/dssg/home/acct-stu/stu464/data/domestic_sound_events/audio/train/weak" "dev/wav.csv"
python extract_feature.py "dev/wav.csv" "dev/feature.h5" --sr 44100 --num_worker 1
ln -s $(realpath "/dssg/home/acct-stu/stu464/data/domestic_sound_events/label/train/weak.csv") "dev/label.csv"

# evaluation set
echo "Preparing evaluation set"
python prepare_wav_csv.py "/dssg/home/acct-stu/stu464/data/domestic_sound_events/audio/eval" "eval/wav.csv"
python extract_feature.py "eval/wav.csv" "eval/feature.h5" --sr 44100 --num_worker 1
ln -s $(realpath "/dssg/home/acct-stu/stu464/data/domestic_sound_events/label/eval/eval.csv") "eval/label.csv"

cp "/dssg/home/acct-stu/stu464/data/domestic_sound_events/label/class_label_indices.txt" "metadata/class_label_indices.txt"
