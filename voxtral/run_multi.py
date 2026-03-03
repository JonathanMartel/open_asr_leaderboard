#!/usr/bin/env python3
"""
run_multi.py - Évalue un modèle Voxtral sur plusieurs datasets en une seule session.

Le modèle est chargé une seule fois, ce qui évite le rechargement coûteux (~5-10 min)
entre chaque dataset. Le sous-échantillonnage est appliqué avant le décodage audio
et la normalisation texte pour minimiser le travail inutile.

Usage:
    python run_multi.py \
        --model_id /scratch/jmartel/models/Voxtral-Small-24B-GPTQ-W4A16 \
        --max_eval_samples 20 \
        --batch_size 8
"""

import argparse
import os
import sys
import time

_MULTI_START = time.time()

def _mlog(msg):
    elapsed = time.time() - _MULTI_START
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"[{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)

_mlog("=== run_multi.py démarré")
_mlog(f"  Python : {sys.version.split()[0]}")
_mlog(f"  PID : {os.getpid()}")

# Garantir le mode offline même sans le shell script SLURM
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_mlog("  Imports principaux (torch, transformers, datasets)...")
_t = time.time()
# Import depuis run_eval.py dans le même dossier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import load_model_and_processor, load_raw_dataset, evaluate_dataset, _log, _log_step
_mlog(f"  └─ run_eval importé : {time.time() - _t:.1f}s")

# Chemin vers eval_utils pour le score final
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "normalizer"))
_t = time.time()
import eval_utils
_mlog(f"  └─ eval_utils importé : {time.time() - _t:.1f}s")


# Configuration des datasets à évaluer
DATASETS = [
    {
        "dataset_path": "/scratch/jmartel/datasets/fleurs",
        "dataset_name": "fleurs",
        "dataset_config": "fr_fr",
        "split": "test",
    },
    {
        "dataset_path": "/scratch/jmartel/datasets/cv-corpus-24.0-2025-12-05/fr",
        "dataset_name": "common_voice",
        "dataset_config": "fr",
        "split": "test",
    },
    {
        "dataset_path": "/scratch/jmartel/datasets/cv-corpus-24.0-2025-12-05/fr",
        "dataset_name": "common_voice_fr_ca",
        "dataset_config": "fr",
        "split": "validated",
        "tsv_file": "validated.tsv",
        "accent_filter": "Français du Canada",
    },
    {
        "dataset_path": "/scratch/jmartel/datasets/sps-corpus-2.0-2025-12-05-fr",
        "dataset_name": "spontaneous",
        "dataset_config": None,
        "split": "full",
        "tsv_file": "ss-corpus-fr.tsv",
        "audio_dir": "audios",
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation multi-datasets Voxtral (chargement unique du modèle)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Chemin ou identifiant HuggingFace du modèle Voxtral.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="GPU à utiliser : -1 = device_map='auto' (tous les GPUs, défaut).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Taille du batch pour l'inférence.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Nombre maximal d'échantillons par dataset (None = tous).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Nombre maximal de tokens générés par échantillon.",
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    _mlog(f"=== Modèle : {args.model_id}")
    _mlog(f"  max_eval_samples={args.max_eval_samples} | batch_size={args.batch_size} | max_new_tokens={args.max_new_tokens}")

    # Chargement unique du modèle
    t_model = time.time()
    model, processor, infer_device = load_model_and_processor(args.model_id, args.device)
    _mlog(f"  Modèle prêt en {time.time() - t_model:.1f}s")

    # Évaluation séquentielle de chaque dataset
    for i, cfg in enumerate(DATASETS):
        _mlog(f"\n{'='*80}")
        _mlog(f"Dataset {i+1}/{len(DATASETS)} : {cfg['dataset_name']} | split={cfg['split']}")
        _mlog(f"{'='*80}")

        dataset = load_raw_dataset(
            dataset_path=cfg["dataset_path"],
            dataset_config=cfg.get("dataset_config"),
            split=cfg["split"],
            tsv_file=cfg.get("tsv_file"),
            audio_dir=cfg.get("audio_dir", "clips"),
            accent_filter=cfg.get("accent_filter"),
            max_eval_samples=args.max_eval_samples,
        )

        evaluate_dataset(
            model=model,
            processor=processor,
            infer_device=infer_device,
            dataset=dataset,
            model_id=args.model_id,
            dataset_path=cfg["dataset_path"],
            dataset_name=cfg["dataset_name"],
            split=cfg["split"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    # Score composite final
    _mlog(f"\n{'='*80}")
    _mlog("=== Score composite final")
    _mlog(f"{'='*80}")
    eval_utils.score_results(results_dir, args.model_id)
    _mlog(f"=== run_multi.py terminé — durée totale : {time.time() - _MULTI_START:.1f}s")


if __name__ == "__main__":
    main()
