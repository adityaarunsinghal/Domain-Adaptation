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
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate

python /scratch/as11919/transformers/examples/language-modeling/run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file /scratch/as11919/Domain-Adaptation/movie_roberta/data/full_qa_train.csv \
    --validation_file /scratch/as11919/Domain-Adaptation/movie_roberta/data/full_qa_dev.csv \
    --do_train \
    --do_eval \
    --output_dir /scratch/as11919/Domain-Adaptation/models/movie_roberta

# python $SCRATCH/transformers/examples/language-modeling/run_mlm.py \
#     --model_name_or_path roberta-base \
#     --train_file $SCRATCH/Domain-Adaptation/movie_roberta/data/full_qa_train.csv \
#     --validation_file $SCRATCH/Domain-Adaptation/movie_roberta/data/full_qa_dev.csv \
#     --do_train \
#     --do_eval \
#     --output_dir $SCRATCH/Domain-Adaptation/models/movie_roberta

echo "Done!"
"
