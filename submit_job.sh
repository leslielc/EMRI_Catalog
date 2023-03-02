#!/bin/bash
#PBS -N jobGPU
#PBS -q qgpgpua100
#PBS -l select=1:ncpus=4:mem=10G:ngpus=1  
#PBS -l walltime=05:00:00

module load conda/4.12.0
conda activate few_env
module load gcc/10.2.0
module load cuda/12.0

python /home/ad/burkeol/work/EMRI_Catalog/Explore/mcmc_run.py 
