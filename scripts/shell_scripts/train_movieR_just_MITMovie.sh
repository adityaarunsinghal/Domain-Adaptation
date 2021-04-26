#!/bin/bash -e

#SBATCH --job-name=MovRNER
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movieR_on_MITMovieNER.out
#SBATCH --error=./slurm_logs/%j_movieR_on_MITMovieNER.err
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

  python $SCRATCH/transformers/examples/token-classification/run_ner.py \
  --model_name_or_path thatdramebaazguy/movie-roberta-base \
  --train_file $SCRATCH/Domain-Adaptation/datasets/movies/MIT_movie_NER/dict_structure/plain_training.json \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/MIT_movie_NER/dict_structure/plain_val.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/movieR_on_MITMovieNER/ \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 20 \
  --num_train_epochs 10 \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy steps \
  --save_steps 1000 \
  --eval_steps 500 \
  --logging_first_step \
  --run_name "Testing movieR on MIT_movie_NER - 10 epoch"

echo "Done!"
"


  # python $SCRATCH/Domain-Adaptation/scripts/run_ner_roberta.py \
  # --model_name_or_path $SCRATCH/Domain-Adaptation/models/movie_roberta/movie_roberta_15April2021 \
  # --train_file $SCRATCH/Domain-Adaptation/datasets/movies/MIT_movie_NER/dict_structure/trivia_training_literal.json \
  # --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/MIT_movie_NER/dict_structure/trivia_val_literal.json \
  # --output_dir $SCRATCH/Domain-Adaptation/models/movieR_on_MITMovieNER/ \
  # --do_train \
  # --do_eval \
  # --per_device_train_batch_size 64 \
  # --per_device_eval_batch_size 20 \
  # --num_train_epochs 1 \
  # --overwrite_output_dir \
  # --overwrite_cache \
  # --evaluation_strategy steps \
  # --save_steps 1000 \
  # --eval_steps 500 \
  # --logging_first_step \
  # --run_name "Testing movieR on MIT_movie_NER - 1 epoch"


  #   python $SCRATCH/Domain-Adaptation/scripts/run_ner_roberta.py \
  # --model_name_or_path $SCRATCH/Domain-Adaptation/models/movie_roberta/movie_roberta_15April2021 \
  # --dataset_name conll2003 \
  # --output_dir $SCRATCH/Domain-Adaptation/models/movieR_on_MITMovieNER/ \
  # --do_train \
  # --do_eval \
  # --per_device_train_batch_size 64 \
  # --per_device_eval_batch_size 20 \
  # --num_train_epochs 1 \
  # --overwrite_output_dir \
  # --overwrite_cache \
  # --evaluation_strategy steps \
  # --save_steps 1000 \
  # --eval_steps 500 \
  # --logging_first_step \
  # --run_name "Testing movieR on MIT_movie_NER - 1 epoch"


  # python $SCRATCH/transformers/examples/token-classification/run_ner.py \
  # --model_name_or_path thatdramebaazguy/movie-roberta-base \
  # --train_file $SCRATCH/Domain-Adaptation/datasets/movies/MIT_movie_NER/dict_structure/trivia_training.json \
  # --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/MIT_movie_NER/dict_structure/trivia_val.json \
  # --output_dir $SCRATCH/Domain-Adaptation/models/movieR_on_MITMovieNER/ \
  # --do_train \
  # --do_eval \
  # --per_device_train_batch_size 64 \
  # --per_device_eval_batch_size 20 \
  # --num_train_epochs 10 \
  # --overwrite_output_dir \
  # --overwrite_cache \
  # --evaluation_strategy steps \
  # --save_steps 1000 \
  # --eval_steps 500 \
  # --logging_first_step \
  # --run_name "Testing movieR on MIT_movie_NER - 10 epoch"


# python run_ner_roberta.py \
#   --model_name_or_path roberta-base \
#   --dataset_name conll2003 \
#   --output_dir /Users/aditya/Desktop/trash_model_rob_on_conll \
#   --learning_rate 3e-5 \
#   --do_train \
#   --do_eval


  # python run_ner_roberta.py \
  # --model_name_or_path robert-base \
  # --dataset_name conll2003 \
  # --output_dir $SCRATCH/Domain-Adaptation/models/robBase_on_conll/ \
  # --learning_rate 3e-5 \
  # --do_train \
  # --do_eval


