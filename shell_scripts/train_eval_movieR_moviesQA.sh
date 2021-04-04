#!/bin/bash -e

#SBATCH --job-name=DAPTeval
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movieR_eval.out
#SBATCH --error=./slurm_logs/%j_movieR_eval.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate

# squadv1 trial 

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path roberta-base \
  --dataset_name squad \
  --validation_file $SCRATCH/Domain-Adaptation/data/squad.film.all.json \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir $SCRATCH/Domain-Adaptation/models/movie_roberta/eval_on_moviesQA/roberta_base_plain_squadv1

# squad v2 gives error because of "no ans probability" 

# python $SCRATCH/transformers/examples/question-answering/run_qa.py \
#   --model_name_or_path roberta-base \
#   --dataset_name squad_v2 \
#   --validation_file $SCRATCH/Domain-Adaptation/data/squad.film.train.json \
#   --do_train \
#   --do_eval \
#   --version_2_with_negative True \
#   --per_device_train_batch_size 12 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir $SCRATCH/Domain-Adaptation/models/movie_roberta/eval_on_moviesQA/roberta_base_plain

echo "Done!"
"