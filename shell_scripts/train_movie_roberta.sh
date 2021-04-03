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

# this was mlm continued on roberta

python $SCRATCH/Domain-Adaptation/make_movies_dataset.py

python /scratch/as11919/transformers/examples/language-modeling/run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file $SCRATCH/Domain-Adaptation/data/simple_movie_text_MLM/ALL_DATA_train.txt \
    --validation_file $SCRATCH/Domain-Adaptation/data/simple_movie_text_MLM/ALL_DATA_test.txt \
    --do_train \
    --do_eval \
    --line_by_line \
    --output_dir $SCRATCH/Domain-Adaptation/movie_roberta/roberta_DAPT_movies_model_withEVAL

echo "Done!"
"

# default_metrics =   "attention_probs_dropout_prob": 0.1,
#   "bos_token_id": 0,
#   "eos_token_id": 2,
#   "gradient_checkpointing": false,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-05,
#   "max_position_embeddings": 514,
#   "model_type": "roberta",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 12,
#   "pad_token_id": 1,
#   "position_embedding_type": "absolute",
#   "transformers_version": "4.4.0.dev0",
#   "type_vocab_size": 1,
#   "use_cache": true,
#   "vocab_size": 50265


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