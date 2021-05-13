#!/bin/bash -e

#SBATCH --job-name=movieQs
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_movieQA_evals.out
#SBATCH --error=./slurm_logs/%j_movieQA_evals.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=0-5:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/roberta_NER_squad/train_1000_egs/NER_trained_for_3_epochs \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase-3epoch \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval RobBase - NER (3 epoch) - squad (1 epoch) - on moviesQA'

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/roberta_NER_squad/train_1000_egs/NER_trained_for_2_epochs \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase-2epoch \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval RobBase - NER (2 epoch) - squad (1 epoch) - on moviesQA'


python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/roberta_NER_squad/train_1000_egs/ \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase-1epoch \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval RobBase - NER (1 epoch) - squad (1 epoch) - on moviesQA'

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad/checkpoint-2768 \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR/1epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval movieR - squad (1 epoch) - on moviesQA'

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad/checkpoint-4152 \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR/2epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval movieR - squad (2 epoch) - on moviesQA'

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad/ \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR/3epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval movieR - squad (3 epoch) - on moviesQA'


  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/roberta_on_NER_on_squad/checkpoint-11072 \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase_NER_squad/1epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval Roberta - NER (2 epoch) - squad (1 epoch) - on moviesQA'


    python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/roberta_on_NER_on_squad/checkpoint-16608 \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase_NER_squad/2epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval Roberta - NER (2 epoch) - squad (2 epoch) - on moviesQA'


    python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/roberta_on_NER_on_squad/ \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase_NER_squad/3epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval Roberta - NER (2 epoch) - squad (3 epoch) - on moviesQA'

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad/checkpoint-2768 \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR_NER_squad/1epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval MovieR - NER (2 epoch) - squad (1 epoch) - on moviesQA'

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad/checkpoint-4152 \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR_NER_squad/2epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval MovieR - NER (2 epoch) - squad (2 epoch) - on moviesQA'


python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/three_epochs/movieR_on_squad/ \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR_NER_squad/3epoch_onsquad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval MovieR - NER (2 epoch) - squad (3 epoch) - on moviesQA'


  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/roberta_5NER_1squad \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/roberta_NER_squad/5NER_1squad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval Roberta - NER (5 epoch) - squad (1 epoch) - on moviesQA'

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path $SCRATCH/Domain-Adaptation/models/movieR_5NER_1squad \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.nested.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/movieR_NER_squad/5NER_1squad \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name 'Eval MovieR - NER (5 epoch) - squad (1 epoch) - on moviesQA'
"
