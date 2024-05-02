#!/bin/bash
#SBATCH -p gypsum-m40
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

python3 800kSpecterEmbeds.py