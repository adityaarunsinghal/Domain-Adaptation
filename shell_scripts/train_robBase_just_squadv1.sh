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

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path roberta-base \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --output_dir $SCRATCH/Domain-Adaptation/models/movie_roberta/eval_on_squadv1/plain_roberta_8april2021

echo "Done! This was the plain roberta base model trained on squadv1"
"
