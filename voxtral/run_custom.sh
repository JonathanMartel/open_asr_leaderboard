#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=def-jmartel
#SBATCH --output=/scratch/jmartel/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/jmartel/logs/slurm-%x-%j.err
#SBATCH --mail-user=martel.jonathan@uqam.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00

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
    "/scratch/jmartel/models/Voxtral-Small-24B-2507"
)

BATCH_SIZE=8

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    # --- Google FLEURS (fr_fr) ---
    python /scratch/jmartel/open_asr_leaderboard/voxtral/run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="/scratch/jmartel/datasets/fleurs" \
        --dataset="fleurs" \
        --dataset_config="fr_fr" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # --- Mozilla Common Voice 24.0 (fr) ---
    python /scratch/jmartel/open_asr_leaderboard/voxtral/run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="/scratch/jmartel/datasets/cv-corpus-24.0-2025-12-05/fr" \
        --dataset="common_voice" \
        --dataset_config="fr" \
        --split="test" \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=200

    # --- Mozilla Common Voice 24.0 - Français du Canada (validated) ---
    python /scratch/jmartel/open_asr_leaderboard/voxtral/run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="/scratch/jmartel/datasets/cv-corpus-24.0-2025-12-05/fr" \
        --dataset="common_voice_fr_ca" \
        --tsv_file="validated.tsv" \
        --accent_filter="Français du Canada" \
        --split="validated" \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=200

    # --- Common Voice Spontaneous 2.0 (fr) ---
    python /scratch/jmartel/open_asr_leaderboard/voxtral/run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="/scratch/jmartel/datasets/sps-corpus-2.0-2025-12-05-fr" \
        --dataset="spontaneous" \
        --tsv_file="ss-corpus-fr.tsv" \
        --audio_dir="audios" \
        --split="full" \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=200
        

    # Evaluate results
    RUNDIR=`pwd` && \
    cd /scratch/jmartel/open_asr_leaderboard/normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done
