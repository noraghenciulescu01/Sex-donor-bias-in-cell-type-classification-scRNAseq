#!/bin/sh
#SBATCH --partition=general   # Request partition. Default is 'general' 
#SBATCH --qos=short           # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=0:40:00        # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1            # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2     # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=20GB             # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=FAIL       # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm_%j.err  # Set name of error log. %j is the Slurm jobId

# which python 1>&2  # Write path to Python binary to standard error
# python --version   # Write Python version to standard error

srun python3 knn_0.py
