python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path /scratch/as11919/Domain-Adaptation/models/three_epochs/roberta_on_squad/checkpoint-2768 \
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
  --run_name "Eval robBase - squad [1 epoch] - on moviesQA"

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path /scratch/as11919/Domain-Adaptation/models/three_epochs/roberta_on_squad/checkpoint-5536 \
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
  --run_name "Eval robBase - squad [2 epoch] - on moviesQA"

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path /scratch/as11919/Domain-Adaptation/models/three_epochs/roberta_on_squad \
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
  --run_name "Eval robBase - squad [3 epoch] - on moviesQA"

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path /scratch/as11919/Domain-Adaptation/models/three_epochs/roberta_on_squad/checkpoint-2768 \
  --dataset_name squad \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_squadv1/robBase-1epoch \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval robBase - squad [1 epoch] - on SQuAD"

  python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path /scratch/as11919/Domain-Adaptation/models/three_epochs/roberta_on_squad/checkpoint-5536 \
  --dataset_name squad \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_squadv1/robBase-2epoch \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval robBase - squad [2 epoch] - on SQuAD"