#!/bin/bash
#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"


module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt # install Python dependencies

export TOKENIZERS_PARALLELISM=false
# runs your code
#distilbert-base-uncased
#SBATCH --job-name=myjob
#SBATCH --array=1-8
#SBATCH --output=myjob_%a.out

srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 5e-4 --num_epochs 5 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 1e-3 --num_epochs 5 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 1e-4 --num_epochs 7 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 5e-4 --num_epochs 7 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 1e-3 --num_epochs 7 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 1e-4 --num_epochs 9 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 5e-4 --num_epochs 9 --small_subset $SLURM_ARRAY_TASK_ID
srun python classificationt5.py --experiment "overfit" --device cuda --model "t5-base" --batch_size "32" --lr 1e-3 --num_epochs 9 --small_subset $SLURM_ARRAY_TASK_ID
