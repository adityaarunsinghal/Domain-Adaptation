#!/bin/bash -e

#SBATCH --job-name=RobNER
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_roberta_on_MITMovieNER.out
#SBATCH --error=./slurm_logs/%j_roberta_on_MITMovieNER.err
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

python $SCRATCH/transformers/examples/token-classification/run_ner.py \
  --model_name_or_path roberta-base \
  --train_file $SCRATCH/Domain-Adaptation/data/MIT_movie_NER/csv_format/trivia_training.csv \
  --validation_file $SCRATCH/Domain-Adaptation/data/MIT_movie_NER/csv_format/trivia_testing.csv \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 512 \
  --overwrite_output_dir \
  --output_dir $SCRATCH/Domain-Adaptation/models/roberta_base_on_MITMovie/

echo "Done! This was the plain roberta base model trained on MITMovie - NER"
"
