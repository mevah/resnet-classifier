#!/bin/bash
#SBATCH  --output=./output
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G

source /itet-stor/himeva/net_scratch/anaconda3/etc/profile.d/conda.sh
conda activate sep
python train.py -lr 0.005 --num-epochs 50 --batch-size 32 --save-every 5 --tensorboard-vis --print-summary 
