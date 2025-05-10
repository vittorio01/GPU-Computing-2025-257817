#!/bin/bash
#SBATCH --job-name=smm-csr-parallel-2
#SBATCH --output=out/output_%j.out
#SBATCH --error=out/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load CUDA/12.5
./bin/smm-csr-parallel-2 ../data/$0