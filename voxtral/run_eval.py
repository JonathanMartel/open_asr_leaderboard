import argparse
import os
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import evaluate
import pandas as pd
from datasets import load_dataset, load_from_disk, Audio, Dataset
from normalizer import data_utils
import time
from tqdm import tqdm

os.environ['HF_EVALUATE_OFFLINE'] = '1'

wer_metric = evaluate.load("/scratch/jmartel/evaluate/metrics/wer/wer.py")

def main(args):
    # Load Voxtral model using transformers
    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.device}" if args.device >= 0 else "cpu",
    )
    model.eval()

    def benchmark(batch):
        # Load audio inputs
        audios = [audio["array"] for audio in batch["audio"]]
        batch["audio_length_s"] = [len(audio) / batch["audio"][0]["sampling_rate"] for audio in audios]
        minibatch_size = len(audios)

        # START TIMING
        start_time = time.time()

        # INFERENCE
        # Process audio inputs for transcription
        pred_text = []
        for audio in audios:
            inputs = processor.apply_transcription_request(
                language="fr",
                audio=[audio],
                sampling_rate=16000,
                format=["wav"],
                model_id=args.model_id,
            )
            inputs = inputs.to(model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            decoded_output = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            pred_text.append(decoded_output[0])

        # END TIMING
        runtime = time.time() - start_time

        # normalize by minibatch size since we want the per-sample time
        batch["transcription_time_s"] = minibatch_size * [runtime / minibatch_size]

        # normalize transcriptions with multilingual normalizer (French)
        batch["predictions"] = [data_utils.ml_normalizer(pred) for pred in pred_text]
        batch["references"] = batch["norm_text"]
        return batch


    cv_tsv = os.path.join(args.dataset_path, f"{args.split}.tsv")
    if os.path.exists(cv_tsv):
        # Format brut Mozilla Common Voice (TSV + clips/)
        clips_dir = os.path.join(args.dataset_path, "clips")
        df = pd.read_csv(cv_tsv, sep="\t", usecols=["path", "sentence"])
        df["audio"] = clips_dir + os.sep + df["path"]
        df = df.drop(columns=["path"])
        ds = Dataset.from_pandas(df, preserve_index=False)
    else:
        ds = load_dataset(args.dataset_path, args.dataset_config, split=args.split, trust_remote_code=True)
    print(ds)

    dataset = ds.cast_column("audio", Audio(sampling_rate=16000))

    def normalize_fr(batch):
        batch["original_text"] = data_utils.get_text(batch)
        batch["norm_text"] = data_utils.ml_normalizer(batch["original_text"])
        return batch

    dataset = dataset.map(normalize_fr, writer_batch_size=1)
    dataset = dataset.filter(data_utils.is_target_text_in_range, input_columns=["norm_text"], writer_batch_size=1)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    print(dataset)

    dataset = dataset.map(
        benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest results (WER and RTFX)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


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
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on.",
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
