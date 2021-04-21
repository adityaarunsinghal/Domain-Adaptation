#!/bin/bash -e

#SBATCH --job-name=RobSquad
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_roberta_on_squadv1.out
#SBATCH --error=./slurm_logs/%j_roberta_on_squadv1.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path roberta-base \
  --dataset_name squad \
  --output_dir $SCRATCH/Domain-Adaptation/models/plain_roberta_on_squadv1 \
  --do_train \
  --do_eval \
  --num_train_epochs 50 \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 7500 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 20 \
  --logging_first_step \
  --max_seq_length 384 \
  --doc_stride 128 \
  --run_name "Training RobBase on Squadv1 - 50 epochs"

echo "Done! This was the plain roberta base model trained on squadv1"
"
