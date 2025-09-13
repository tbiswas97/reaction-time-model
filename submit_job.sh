#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name k2
#SBATCH --output  /gs/gsfs0/users/tbiswas/bayes_seg/logs/job.%J.out
#SBATCH --error   /gs/gsfs0/users/tbiswas/bayes_seg/logs/job.%J.err
#SBATCH --array=1-17


# start from launch dir
cd $SLURM_SUBMIT_DIR

# activate conda environment 
conda init 
conda activate seg

# run jobs 
CMDFILE=commands_k2.txt
CMD=$(awk "NR==$SLURM_ARRAY_TASK_ID" $CMDFILE)
$CMD
