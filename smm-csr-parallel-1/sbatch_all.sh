#!/bin/bash
#SBATCH --job-name=smm-csr-parallel-1
#SBATCH --output=out/output_%j.out
#SBATCH --error=out/error_%j.err
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load CUDA/12.3.2
nvidia-smi
echo " "
echo "---------- Processing matrices  ----------"
./bin/smm-csr-parallel-1 ../data/dbir2.mtx $1 $2 $3 $4
./bin/smm-csr-parallel-1 ../data/ifiss_mat.mtx $1 $2 $3 $4
./bin/smm-csr-parallel-1 ../data/ex11.mtx $1 $2 $3 $4
./bin/smm-csr-parallel-1 ../data/language.mtx $1 $2 $3 $4
./bin/smm-csr-parallel-1 ../data/Linux_call_graph.mtx $1 $2 $3 $4
./bin/smm-csr-parallel-1 ../data/nemeth24.mtx $1 $2 $3 $4
./bin/smm-csr-parallel-1 ../data/twotone.mtx $1 $2 $3 $4
echo " "
