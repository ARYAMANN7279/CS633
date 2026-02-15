#!/bin/bash
#SBATCH -J Ass1_P8
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:10:00
#SBATCH -o logs/P8_%j.slurm.out
#SBATCH -e logs/P8_%j.slurm.err

set -euo pipefail

module purge
module load compiler/oneapi-2024/mpi/2021.12

# If required on your system
export I_MPI_PMI_LIBRARY=/lib64/libpmi2.so

# Problem parameters
D1=2
D2=4
T=10
SEED=1000

M_LIST=(262144 1048576)
REPEAT=5

CSV="results/times_P8.csv"
echo "P,M,run,maxD1,maxD2,time" > "$CSV"

for M in "${M_LIST[@]}"; do
	for ((run=1; run<=REPEAT; run++)); do
		echo "P=8 run=$run M=$M"
		line=$(srun --mpi=pmi2 ./src $M $D1 $D2 $T $SEED | tr ' ' ',')
    	 	echo "8,$M,$run,$line" >> $CSV
     	done
done
