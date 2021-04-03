#!/bin/bash -e

#SBATCH --job-name=movieRoberta
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movie_roberta.out
#SBATCH --error=./slurm_logs/%j_movie_roberta.err
#SBATCH --export=ALL
# #SBATCH --cpus-per-task=8 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate

# this was mlm continued on roberta

export TRAIN_FILE=/scratch/as11919/Domain-Adaptation/data/simple_movie_text_MLM/ALL_DATA_train.txt
export TEST_FILE=/scratch/as11919/Domain-Adaptation/data/simple_movie_text_MLM/ALL_DATA_test.txt

python run_language_modeling_HF.py \
    --output_dir=/scratch/as11919/Domain-Adaptation/movie_roberta/roberta_DAPT_movies_model \
    --model_type=roberta \
    --model_name_or_path=roberta-base \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm

echo "Done!"
"

# this was movie_roberta from scratch
# python DAPT_movies_scratch.py

# this was wiki_movies roberta
# python /scratch/as11919/transformers/examples/language-modeling/run_mlm.py \
#     --model_name_or_path roberta-base \
#     --train_file /scratch/as11919/Domain-Adaptation/movie_roberta/data/full_qa_train.csv \
#     --validation_file /scratch/as11919/Domain-Adaptation/movie_roberta/data/full_qa_dev.csv \
#     --do_train \
#     --do_eval \
#     --output_dir /scratch/as11919/Domain-Adaptation/models/movie_roberta