#!/bin/bash

#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a 

python Amazon_sentiment_combined.py