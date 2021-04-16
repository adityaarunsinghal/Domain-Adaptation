#!/bin/bash -e

#SBATCH --job-name=debug
#SBATCH --nodes=1
#SBATCH --output=./slurm_logs/%j.out
#SBATCH --error=./slurm_logs/%j.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -c 8

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate

python /scratch/as11919/transformers/examples/language-modeling/run_mlm.py \
    --model_name_or_path distilroberta-base \
    --train_file /scratch/as11919/Domain-Adaptation/data/simple_movie_text_MLM/movie_names_25mlens_small_debug.txt \
    --validation_file /scratch/as11919/Domain-Adaptation/data/simple_movie_text_MLM/movie_names_25mlens_small_debug.txt \
    --do_train \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 1 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --save_steps 500 \
    --eval_steps 500 \
    --line_by_line \
    --output_dir /scratch/as11919/Domain-Adaptation/movie_roberta/to_delete \
    --logging_first_step \
    --overwrite_output_dir \
    --run_name 'debugging sh file'

echo "Done!"
"