#!/bin/bash
#SBATCH --job-name=calphy-rootstock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="intel&gpu40"
#SBATCH --time=02:00:00

module purge
module load anaconda3/2024.10
module load gcc-toolset/14 openmpi/gcc/4.1.8

conda activate calphy

calphy -i input.yaml
