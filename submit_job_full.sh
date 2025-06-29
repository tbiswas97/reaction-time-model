#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name seg_0
#SBATCH --output  /gs/gsfs0/users/tbiswas/bayes_seg/logs/job.%J.out
#SBATCH --error   /gs/gsfs0/users/tbiswas/bayes_seg/logs/job.%J.err
#SBATCH --array=1-47


# start from launch dir
cd $SLURM_SUBMIT_DIR

# activate conda environment 
conda init 
conda activate seg

# run jobs 
CMDFILE=commands_full.txt
CMD=$(awk "NR==$SLURM_ARRAY_TASK_ID" $CMDFILE)
$CMD
