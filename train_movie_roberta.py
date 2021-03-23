import transformers
from datasets import load_dataset
import 

dataset = load_dataset('csv', data_files={'train' : 'movie_roberta/data/full_qa_train.csv',
'val' : 'movie_roberta/data/full_qa_dev.csv',
'test' : 'movie_roberta/data/full_qa_test.csv'})

python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name dataset \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /scratch/as11919/Domain-Adaptation/models/movie_roberta