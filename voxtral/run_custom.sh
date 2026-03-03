#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=def-jmartel
#SBATCH --output=/scratch/jmartel/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/jmartel/logs/slurm-%x-%j.err
#SBATCH --mail-user=martel.jonathan@uqam.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=8:00:00

module --force purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 cuda/12.9 python/3.13.2 \
            opencv/4.13.0 ffmpeg/7.1.1 arrow/22
source $SCRATCH/venv_asr_test/bin/activate


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export PYTHONPATH="..":$PYTHONPATH

MODEL_IDs=(
    "/scratch/jmartel/models/Voxtral-Small-24B-GPTQ-W4A16"
    # Voxtral BF16 (non-quantifié) : ~48 Go de poids → OOM sur 2×A100 40Go
    # Nécessite a100:4 ou des A100 80Go pour l'inférence
     "/scratch/jmartel/models/Voxtral-Small-24B-2507"
)

BATCH_SIZE=8

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    # Évaluation multi-datasets : le modèle est chargé une seule fois
    # puis évalué successivement sur FLEURS, CV, CV fr-CA et spontané.
    python /scratch/jmartel/open_asr_leaderboard/voxtral/run_multi.py \
        --model_id="${MODEL_ID}" \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

done
