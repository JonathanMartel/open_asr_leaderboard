import argparse
import os
import time
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import evaluate
import pandas as pd
from datasets import load_dataset, Audio, Dataset, Features, Value
from normalizer import data_utils
from tqdm import tqdm

os.environ['HF_EVALUATE_OFFLINE'] = '1'

# ─── Helpers de timing ────────────────────────────────────────────────────────

_SCRIPT_START = time.time()

def _ts():
    """Retourne le temps écoulé depuis le démarrage du script (HH:MM:SS)."""
    elapsed = time.time() - _SCRIPT_START
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _log(msg):
    print(f"[{_ts()}] {msg}", flush=True)

def _log_step(label, t0):
    elapsed = time.time() - t0
    print(f"[{_ts()}]   └─ {label} : {elapsed:.1f}s", flush=True)
    return time.time()

def _log_gpu():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserv = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            _log(f"  GPU {i} ({torch.cuda.get_device_name(i)}) : "
                 f"alloué={alloc:.1f}Go / réservé={reserv:.1f}Go / total={total:.1f}Go")

# ─── Chargement métrique WER ──────────────────────────────────────────────────

_log("Chargement de la métrique WER...")
_t = time.time()
wer_metric = evaluate.load("/scratch/jmartel/evaluate/metrics/wer/wer.py")
_log_step("evaluate.load(wer)", _t)


# ─── Fonctions principales ────────────────────────────────────────────────────

def load_model_and_processor(model_id, device=-1):
    """Charge le modèle Voxtral et son processor sur le(s) GPU(s) disponibles."""
    _log(f"=== Chargement du modèle : {model_id}")
    t_total = time.time()

    _log("  Chargement du processor...")
    t = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    _log_step("AutoProcessor.from_pretrained", t)

    device_map = "auto" if device < 0 else f"cuda:{device}"
    _log(f"  Chargement des poids du modèle (device_map={device_map!r})...")
    t = time.time()
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    _log_step("VoxtralForConditionalGeneration.from_pretrained", t)

    t = time.time()
    model.eval()
    _log_step("model.eval()", t)

    # Résoudre le device cible (device_map="auto" rend model.device ambigu)
    infer_device = next(model.parameters()).device
    _log(f"  Device d'inférence : {infer_device}")
    _log_gpu()
    _log_step("Chargement modèle total", t_total)

    return model, processor, infer_device


def load_raw_dataset(dataset_path, dataset_config="fr_fr", split="test",
                     tsv_file=None, audio_dir="clips", accent_filter=None,
                     max_eval_samples=None):
    """
    Charge le dataset brut (TSV ou HuggingFace), applique les filtres de base
    et le sous-échantillonnage AVANT toute opération coûteuse (décodage audio,
    normalisation texte), puis prépare le dataset pour l'inférence.
    """
    _log(f"--- Chargement dataset : {dataset_path} | split={split}")
    t_total = time.time()

    tsv_filename = tsv_file if tsv_file else f"{split}.tsv"
    tsv_path = os.path.join(dataset_path, tsv_filename)

    if os.path.exists(tsv_path):
        _log(f"  Format TSV détecté : {tsv_path}")
        t = time.time()
        audio_full_dir = os.path.join(dataset_path, audio_dir)
        df = pd.read_csv(tsv_path, sep="\t")
        _log_step(f"pd.read_csv ({len(df)} lignes)", t)

        # Détection automatique de la colonne audio (CV: "path", spontané: "audio_file")
        audio_col = next((c for c in ["path", "audio_file"] if c in df.columns), None)
        if audio_col is None:
            raise ValueError(f"Aucune colonne audio trouvée. Colonnes disponibles : {list(df.columns)}")
        df["audio"] = audio_full_dir + os.sep + df[audio_col]

        # Filtrage par accent avant sous-échantillonnage
        if accent_filter and "accents" in df.columns:
            before = len(df)
            df = df[df["accents"] == accent_filter]
            _log(f"  Filtrage accent '{accent_filter}' : {before} → {len(df)} entrées")

        # Sous-échantillonnage sur le DataFrame (avant création du Dataset)
        if max_eval_samples is not None and max_eval_samples > 0:
            _log(f"  Sous-échantillonnage : {len(df)} → {min(max_eval_samples, len(df))} échantillons")
            df = df.head(max_eval_samples)

        t = time.time()
        text_cols = [c for c in ["sentence", "transcription", "text", "normalized_text", "transcript"] if c in df.columns]
        # Créer avec features explicites : Audio() évite l'erreur "large_string → struct" de PyArrow
        features = Features({"audio": Audio(), **{col: Value("string") for col in text_cols}})
        ds_dict = {"audio": df["audio"].tolist(), **{col: df[col].fillna("").tolist() for col in text_cols}}
        ds = Dataset.from_dict(ds_dict, features=features)
        _log_step(f"Dataset.from_dict ({len(ds)} samples)", t)

    else:
        _log(f"  Format HuggingFace détecté : {dataset_path} (config={dataset_config})")
        t = time.time()
        ds = load_dataset(dataset_path, dataset_config, split=split, trust_remote_code=True)
        _log_step(f"load_dataset ({len(ds)} samples)", t)

        # Sous-échantillonnage HF avant opérations coûteuses
        if max_eval_samples is not None and max_eval_samples > 0:
            n = min(max_eval_samples, len(ds))
            _log(f"  Sous-échantillonnage : {len(ds)} → {n} échantillons")
            t = time.time()
            ds = ds.select(range(n))
            _log_step("ds.select", t)

    _log(f"  Dataset brut : {ds}")

    # cast_column : définit le pipeline de décodage audio (lazy, pas de décodage immédiat)
    _log("  cast_column → Audio(16kHz)...")
    t = time.time()
    dataset = ds.cast_column("audio", Audio(sampling_rate=16000))
    _log_step("cast_column (lazy)", t)

    def normalize_fr(batch):
        batch["original_text"] = data_utils.get_text(batch)
        batch["norm_text"] = data_utils.ml_normalizer(batch["original_text"])
        return batch

    _log(f"  map(normalize_fr) sur {len(dataset)} samples...")
    t = time.time()
    dataset = dataset.map(normalize_fr, writer_batch_size=1)
    _log_step(f"map(normalize_fr) — {len(dataset)} samples traités", t)

    _log("  filter(is_target_text_in_range)...")
    t = time.time()
    n_before = len(dataset)
    dataset = dataset.filter(data_utils.is_target_text_in_range, input_columns=["norm_text"], writer_batch_size=1)
    _log_step(f"filter : {n_before} → {len(dataset)} samples", t)

    _log_step("load_raw_dataset total", t_total)
    _log(f"  Dataset final : {dataset}")
    return dataset


def evaluate_dataset(model, processor, infer_device, dataset, model_id,
                     dataset_path, dataset_name, split, batch_size=8, max_new_tokens=500):
    """
    Lance l'inférence Voxtral sur le dataset et retourne les résultats.
    Écrit le manifest JSONL et affiche WER/RTFx.
    """
    _log(f"--- Inférence : {dataset_name} | {len(dataset)} samples | batch_size={batch_size}")
    t_eval_total = time.time()

    sample_count = [0]  # compteur mutable pour la closure

    def benchmark(batch):
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        t_batch = time.time()
        pred_text = []

        for idx, audio in enumerate(audios):
            sample_idx = sample_count[0]
            sample_count[0] += 1

            # apply_transcription_request
            t_step = time.time()
            inputs = processor.apply_transcription_request(
                language="fr",
                audio=[audio],
                sampling_rate=16000,
                format=["wav"],
                model_id=model_id,
            )
            t_prep = time.time() - t_step

            # Transfert vers GPU
            t_step = time.time()
            inputs = inputs.to(infer_device, dtype=torch.bfloat16)
            t_to = time.time() - t_step

            # Génération
            t_step = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            t_gen = time.time() - t_step

            decoded_output = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            pred_text.append(decoded_output[0])

            audio_dur = batch["audio_length_s"][idx]
            _log(f"  Sample {sample_idx:04d} | audio={audio_dur:.1f}s | "
                 f"prep={t_prep:.2f}s | to_gpu={t_to:.3f}s | generate={t_gen:.2f}s | "
                 f"RTFx={audio_dur / t_gen:.1f}")

        runtime = time.time() - t_batch
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]
        batch["predictions"] = [data_utils.ml_normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch

    _log("  Lancement de dataset.map(benchmark)...")
    t = time.time()
    dataset = dataset.map(
        benchmark, batch_size=batch_size, batched=True, remove_columns=["audio"],
    )
    _log_step(f"dataset.map(benchmark) — {len(dataset)} samples", t)

    _log("  Collecte des résultats...")
    t = time.time()
    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])
    _log_step("Collecte résultats", t)

    t = time.time()
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        model_id,
        dataset_path,
        dataset_name,
        split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    _log_step("write_manifest", t)
    _log(f"  Résultats sauvegardés : {os.path.abspath(manifest_path)}")

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    _log(f"  WER={wer}% | RTFx={rtfx}")
    _log_step(f"evaluate_dataset total ({dataset_name})", t_eval_total)

    return all_results


def main(args):
    model, processor, infer_device = load_model_and_processor(args.model_id, args.device)

    dataset = load_raw_dataset(
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.split,
        tsv_file=args.tsv_file,
        audio_dir=args.audio_dir,
        accent_filter=args.accent_filter,
        max_eval_samples=args.max_eval_samples,
    )

    evaluate_dataset(
        model=model,
        processor=processor,
        infer_device=infer_device,
        dataset=dataset,
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with transformers (e.g., 'mistralai/Voxtral-Mini-3B-2507')",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="esb/datasets",
        help="Dataset path. By default, it is `esb/datasets`",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name. *E.g.* `'fleurs'` for Google Fleurs, or `'common_voice'` for Common Voice.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="fr_fr",
        help="Dataset configuration name. *E.g.* `'fr_fr'` for Fleurs French, or `'fr'` for Common Voice French.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of the dataset. *E.g.* `'validation`' for the dev split, or `'test'` for the test split.",
    )
    parser.add_argument(
        "--tsv_file",
        type=str,
        default=None,
        help="Nom du fichier TSV pour les corpus bruts. Par défaut : '{split}.tsv'. "
             "Utiliser pour les corpus non-standard (ex: 'ss-corpus-fr.tsv').",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="clips",
        help="Sous-dossier contenant les fichiers audio. Défaut: 'clips' (Common Voice). "
             "Utiliser 'audios' pour le corpus spontané.",
    )
    parser.add_argument(
        "--accent_filter",
        type=str,
        default=None,
        help="Filtrer les entrées par valeur de la colonne 'accents' du TSV. "
             "Ex: 'Français du Canada' pour isoler le français canadien.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="GPU à utiliser : -1 = device_map='auto' (tous les GPUs disponibles, défaut), 0 = cuda:0, 1 = cuda:1, etc.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Choose whether you'd like to download the entire dataset or stream it during the evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warm-up steps to run before launching the timed runs.",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
