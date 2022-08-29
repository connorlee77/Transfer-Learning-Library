#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "water-grayscale"   # job name
#SBATCH --mail-user=clee@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source /home/clee/miniconda3/etc/profile.d/conda.sh
conda activate ttlib
CUDA_VISIBLE_DEVICES=0 
python cdan.py /home/carson/features --data=coco-m3fd --source=coco --target=m3fd --arch=resnet50 --batch-size=32 --epochs=30 --workers=8 --log=logs/cdan/coco-m3fd --weighted-sample