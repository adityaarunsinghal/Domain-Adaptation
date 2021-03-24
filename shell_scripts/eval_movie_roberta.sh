#!/bin/bash -e

#SBATCH --job-name=evalMR
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_eval_movie_roberta.out
#SBATCH --error=./slurm_logs/%j_eval_movie_roberta.err
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

# on squad

python /scratch/as11919/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path ~/Domain-Adaptation/models/movie_roberta \
  --dataset_name squad_v2 \
  --do_eval \
  --version_2_with_negative True\
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /scratch/as11919/Domain-Adaptation/models/movie_roberta/eval_on_squadv2

# on MoviesQA

# python $SCRATCH/transformers/examples/question-answering/run_qa.py \
#   --model_name_or_path ~/Domain-Adaptation/models/movie_roberta \
#   --train_file $SCRATCH/Domain-Adaptation/data/squad.film.train.json \
#   --validation_file $SCRATCH/Domain-Adaptation/data/squad.film.dev.json \
#   --do_eval \
#   --do_predict \
#   --version_2_with_negative True\
#   --per_device_train_batch_size 12 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir $SCRATCH/Domain-Adaptation/models/movie_roberta/eval_on_moviesQA

echo "Done!"
"
