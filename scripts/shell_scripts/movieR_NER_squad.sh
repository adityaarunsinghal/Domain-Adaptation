#!/bin/bash -e

#SBATCH --job-name=movNERSquad
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movieR_NER_squad.out
#SBATCH --error=./slurm_logs/%j_movieR_NER_squad.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/movieR_on_MITMovieNER/QA_config_model \
  --dataset_name squad \
  --output_dir $SCRATCH/Domain-Adaptation/models/movieR_NER_squad \
  --do_train \
  --do_eval \
  --num_train_epochs 10 \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 20 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluation_strategy steps \
  --logging_first_step \
  --run_name "MovieR on NER and next on Squadv1 - 10 epochs"

echo "Done!"
"

    python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/movieR_on_MITMovieNER/five_epochs/QA_config_model \
  --dataset_name squad \
  --output_dir $SCRATCH/Domain-Adaptation/models/movieR_5NER_1squad \
  --do_train \
  --do_eval \
  --num_train_epochs 1 \
  --evaluation_strategy steps \
  --eval_steps 1500 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --logging_first_step \
  --run_name "MovieR on NER (5 epoch) and train on Squadv1 - 1 epoch"