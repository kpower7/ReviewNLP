#!/bin/bash

#SBATCH --nodes=10               # Request 2 nodes
#SBATCH --ntasks-per-node=8     # 2 tasks per node
#SBATCH --gres=gpu:1            # Request 2 GPUs per node
#SBATCH --mem=26G               # Memory per node
#SBATCH --time=04:00:00         # Max runtime

source /etc/profile
module load anaconda/2022a 

python Amazon_sentiment_combined_optimized.py