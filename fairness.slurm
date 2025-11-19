#!/bin/bash
#SBATCH --job-name=
#SBATCH -c 1
#SBATCH -t 4:00:00
#SBATCH -o /usr/xtmp/ham51/script-outputs/%j.out
#SBATCH -e /usr/xtmp/ham51/script-outputs/%j.err
#SBATCH --mem=100G
#SBATCH --array=0-107%20

python fairness_batch.py --index $((SLURM_ARRAY_TASK_ID)) --rerun