#!/bin/bash
#SBATCH -J nb_job                
#SBATCH -N 1                     
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=4     
#SBATCH --mem=16G            
#SBATCH -t 24:00:00          
#SBATCH -o slurm-%j.out 



jupyter nbconvert --to notebook --execute mass_tau_6.ipynb \
  --inplace \
  --ExecutePreprocessor.timeout=-1 \
