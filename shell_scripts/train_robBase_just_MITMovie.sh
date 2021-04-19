#!/bin/bash -e

#SBATCH --job-name=RobNER
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_roberta_on_MITMovieNER.out
#SBATCH --error=./slurm_logs/%j_roberta_on_MITMovieNER.err
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
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

python $SCRATCH/transformers/examples/token-classification/run_ner.py \
  --model_name_or_path roberta-base \
  --train_file $SCRATCH/Domain-Adaptation/data/MIT_movie_NER/dict_structure/trivia_training.jsonl \
  --validation_file $SCRATCH/Domain-Adaptation/data/MIT_movie_NER/dict_structure/trivia_val.jsonl \
  --output_dir $SCRATCH/Domain-Adaptation/models/roberta_base_on_MITMovieNER/ \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 20 \
  --num_train_epochs 1 \
  --overwrite_output_dir \
  --evaluation_strategy steps \
  --save_steps 1000 \
  --eval_steps 500 \
  —-preprocessing_num_workers 8 \
  --logging_first_step \
  --run_name "Testing RobBase on MIT_movie_NER"

echo "Done! This was the plain roberta base model trained on MITMovie - NER"
"
