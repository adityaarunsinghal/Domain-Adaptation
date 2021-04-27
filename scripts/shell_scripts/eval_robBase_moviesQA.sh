#!/bin/bash -e

#SBATCH --job-name=robEval
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=./slurm_logs/%j_evaluate_robBase_moviesQA.out
#SBATCH --error=./slurm_logs/%j_evaluate_robBase_moviesQA.err
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH -c 8
#SBATCH --mail-user=adis@nyu.edu

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate

python $SCRATCH/transformers/examples/question-answering/run_qa.py \
  --model_name_or_path thatdramebaazguy/roberta-base-squad \
  --validation_file $SCRATCH/Domain-Adaptation/datasets/movies/squad.film.all.squad_format.json \
  --output_dir $SCRATCH/Domain-Adaptation/models/eval_on_moviesQA/robBase \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 7500 \
  --eval_steps 500 \
  --per_device_eval_batch_size 32 \
  --overwrite_output_dir \
  --overwrite_cache \
  --logging_first_step \
  --run_name "Eval robBase - squad - on moviesQA"

echo "Done!"
"

# {"paragraphs": 
#               [{"context": "Michael Mike Cutter is a fictional character on the long-running NBC series  Law & Order and its spinoff Law & Order: Special Victims Unit played by Linus Roache.", 
#               "qas": 
#                     [{"answers": 
#                                 [{"answer_start": 148, 
#                                   "text": "Linus Roache"}], 
#                       "question": "what actor plays mike cutter on law and order", 
#                       "id": "549698_4"}
#                     ]
#                 }]
# }