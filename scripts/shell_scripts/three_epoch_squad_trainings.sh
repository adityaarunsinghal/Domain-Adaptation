#!/bin/bash -e

#SBATCH --job-name=Squads
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movieR_on_ner_then_squad.out
#SBATCH --error=./slurm_logs/%j_movieR_on_ner_then_squad.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

# python $SCRATCH/transformers/examples/question-answering/run_qa.py \
#   --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/roberta_on_NER/checkpoint-246/QA_config_model \
#   --dataset_name squad \
#   --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
#   --output_dir $SCRATCH/Domain-Adaptation/models/three_epochs/roberta_on_NER_on_squad \
#   --do_train \
#   --do_eval \
#   --num_train_epochs 3 \
#   --save_strategy epoch \
#   --evaluation_strategy epoch \
#   --eval_steps 500 \
#   --per_device_eval_batch_size 32 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --logging_first_step \
#   --run_name 'Roberta Base on NER (2 epoch) and train on Squadv1 - Eval on MoviesQA'

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_NER/checkpoint-124/QA_config_model \
  --dataset_name squad \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_NER_on_squad \
  --do_train \
  --do_eval \
  --num_train_epochs 3 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'MovieR on NER (2 epoch) and train on Squadv1 - Eval on MoviesQA'

# python $SCRATCH/transformers/examples/question-answering/run_qa.py \
#   --model_name_or_path thatdramebaazguy/movie-roberta-base \
#   --dataset_name squad \
#   --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
#   --output_dir $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad \
#   --do_train \
#   --do_eval \
#   --num_train_epochs 3 \
#   --evaluation_strategy epoch \
#   --save_strategy epoch \
#   --eval_steps 7500 \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 20 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --logging_first_step \
#   --run_name 'MovieR-15Apr21 on Squadv1 - 3 epochs - Eval on MoviesQA'


# python $SCRATCH/transformers/examples/question-answering/run_qa.py \
#   --model_name_or_path roberta-base \
#   --dataset_name squad \
#   --output_dir $SCRATCH/Domain-Adaptation/models/three_epochs/roberta_on_squad \
#   --do_train \
#   --do_eval \
#   --num_train_epochs 3 \
#   --evaluation_strategy epoch \
#   --save_strategy epoch \
#   --eval_steps 7500 \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 20 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --overwrite_output_dir \
#   --overwrite_cache \
#   --logging_first_step \
#   --run_name "Roberta Base on Squadv1 - 3 epochs"

"
