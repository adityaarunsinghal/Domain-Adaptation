#!/bin/bash -e

#SBATCH --job-name=robSqMo
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_plainRob_trainSquad_testMovQA.out
#SBATCH --error=./slurm_logs/%j_plainRob_trainSquad_testMovQA.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate

# squadv1 trial with plain roberta

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path roberta-base \
  --train_file /scratch/as11919/Domain-Adaptation/data/plain_squadv1/train-v1.1.json \
  --validation_file /scratch/as11919/Domain-Adaptation/data/plain_squadv1/dev-v1.1.json \
  --test_file $SCRATCH/Domain-Adaptation/data/squad.film.all.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 12 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir \
  --output_dir $SCRATCH/Domain-Adaptation/models/movie_roberta/eval_on_moviesQA/roberta_base_plain_squadv1

echo "Done!"
"
