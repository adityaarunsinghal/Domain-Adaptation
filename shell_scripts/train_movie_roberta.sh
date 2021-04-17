#!/bin/bash -e

#SBATCH --job-name=movieRoberta
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movie_roberta.out
#SBATCH --error=./slurm_logs/%j_movie_roberta.err
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

# there has to be an argument to start with evaluation!!! check perplexity before training

python $SCRATCH/Domain-Adaptation/make_movies_dataset.py

python /scratch/as11919/transformers/examples/language-modeling/run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file $SCRATCH/Domain-Adaptation/data/simple_movie_text_MLM/ALL_DATA_train.txt \
    --validation_file $SCRATCH/Domain-Adaptation/data/simple_movie_text_MLM/ALL_DATA_test.txt \
    --output_dir $SCRATCH/Domain-Adaptation/movie_roberta/movie_roberta_15April2021 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --num_train_epochs 100 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --save_steps 100000 \
    --eval_steps 500 \
    --line_by_line \
    --logging_first_step \
    --overwrite_output_dir \
    --overwrite_cache \
    --run_name "Making MovieR 100 epoch"

echo "Done! this was Making MovieR 100 epoch"
"

# do everything on interactive!!!!
# try distilbert

#also export datsets as jsons directly from HF datasets library

# use Json Lines format!!!!!