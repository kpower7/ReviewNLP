#!/bin/bash

#SBATCH --partition=xeon-g6-volta  # Use GPU-enabled partition
#SBATCH --nodes=4                  # Max allowed for you
#SBATCH --ntasks-per-node=2         # 2 tasks per node
#SBATCH --cpus-per-task=20          # 20 CPUs per task (each node has 40 CPUs)
#SBATCH --gres=gpu:volta:1          # Request 1 GPU per node
#SBATCH --mem=300G                  # Use available memory wisely (each node has ~376.9GB)
#SBATCH --time=08:00:00             # Set max runtime

source /etc/profile
module load anaconda/2022a 

python Amazon_sentiment_optimized_v2.py