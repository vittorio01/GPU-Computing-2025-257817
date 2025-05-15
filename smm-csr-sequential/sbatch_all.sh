#!/bin/bash
#SBATCH --job-name=smm-csr-sequential
#SBATCH --output=out/output_%j.out
#SBATCH --error=out/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module purge
module load CUDA/12.3.2
nvidia-smi
echo " "
echo "---------- Processing matrices  ----------"
./bin/smm-csr-sequential ../data/dbir2.mtx 
./bin/smm-csr-sequential ../data/ifiss_mat.mtx 
./bin/smm-csr-sequential ../data/ex11.mtx 
./bin/smm-csr-sequential ../data/language.mtx
./bin/smm-csr-sequential ../data/Linux_call_graph.mtx
./bin/smm-csr-sequential ../data/nemeth24.mtx 
./bin/smm-csr-sequential ../data/twotone.mtx
echo " "
