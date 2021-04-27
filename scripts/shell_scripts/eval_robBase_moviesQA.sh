#!/bin/bash -e

#SBATCH --job-name=movQA
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_evaluate_moviesQA.out
#SBATCH --error=./slurm_logs/%j_evaluate_moviesQA.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

#robBase

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path thatdramebaazguy/roberta-base-squad \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval robBase - squad - on moviesQA"

# movieR

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path thatdramebaazguy/movie-roberta-squad \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval movieR - squad - on moviesQA"

# movieR-NER-squad

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/movieR_NER_squad/original_QA_config_model \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR_NER_squad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval MovieR - NER - squad - on moviesQA"

# robB-NER-squad

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/roberta_NER_squad/original_QA_config_model \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase_NER_squad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval RobBase - NER - squad - on moviesQA"

echo "Done!"
"
