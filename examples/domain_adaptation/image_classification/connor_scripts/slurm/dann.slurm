#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "dann"   # job name
#SBATCH --mail-user=clee@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source /home/clee/miniconda3/etc/profile.d/conda.sh
conda activate ttlib
CUDA_VISIBLE_DEVICES=0
export UDA_DATA_DIR=/central/scratch/clee/coco_m3fd_uda
bash connor_scripts/bash/dann.bash