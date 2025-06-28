from tqdm import tqdm
import torch
import jiwer
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import login
import datasets as hugDS
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)
from peft import PeftModel, PeftConfig
import numpy as np

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "eval_phowhisper_large_vi_all.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

load_dotenv("./configs/.env")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logging.error("Hugging Face token not found in .env file.")
    raise ValueError(
        "Hugging Face token not found. Please set HF_TOKEN in ./configs/.env"
    )
login(token=hf_token)

cache_dir = os.path.join(os.getcwd(), "cache")
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

JIWER_TRANS = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def load_dataset(region="All"):
    """
    Load and preprocess the ViMD dataset for evaluation.

    Args:
        region (str): Dataset region to filter (e.g., 'Central', 'All' for no filter).

    Returns:
        Dataset: Filtered and preprocessed dataset.
    """
    logging.info(f"Loading dataset with region filter: {region}")
    try:
        dataset = hugDS.load_dataset(
            "nguyendv02/ViMD_Dataset", split="test", cache_dir=cache_dir
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise
    if region != "All":
        dataset = dataset.filter(lambda x: x["region"] == region)
    dataset = dataset.cast_column("audio", hugDS.Audio(sampling_rate=16000))
    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Dataset features: {dataset.features}")

    def filter_long_samples(sample):
        """
        Filter out samples longer than 30 seconds.
        """
        return sample["audio"]["array"].shape[0] <= 30 * 16000

    dataset = dataset.filter(filter_long_samples)
    logging.info(f"Filtered dataset size: {len(dataset)}")

    return dataset


PEFT_MODEL_ID = "minhtien2405/phowhisper-large-all-vi"
try:
    BASE_MODEL_ID = PeftConfig.from_pretrained(
        PEFT_MODEL_ID, cache_dir=cache_dir
    ).base_model_name_or_path
    logging.info(f"Loading model from {BASE_MODEL_ID} with PEFT model {PEFT_MODEL_ID}")
except Exception as e:
    logging.error(f"Failed to load PEFT config: {str(e)}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

try:
    FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(
        BASE_MODEL_ID, cache_dir=cache_dir
    )
    TOKENIZER = WhisperTokenizer.from_pretrained(
        BASE_MODEL_ID, cache_dir=cache_dir, language="vi", task="transcribe"
    )
    BASE_MODEL = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    BASE_MODEL.config.forced_decoder_ids = TOKENIZER.get_decoder_prompt_ids(
        language="vi", task="transcribe"
    )
    logging.info("Feature extractor, tokenizer, and base model loaded successfully.")

    MODEL = PeftModel.from_pretrained(
        BASE_MODEL, PEFT_MODEL_ID, cache_dir=cache_dir
    ).merge_and_unload(
        progressbar=True
    )  # Merge LoRA weights to reduce inference latency
except Exception as e:
    logging.error(f"Failed to load model or components: {str(e)}")
    raise

PIPE = AutomaticSpeechRecognitionPipeline(
    model=MODEL,
    tokenizer=TOKENIZER,
    feature_extractor=FEATURE_EXTRACTOR,
    return_timestamps=True,  # temp
)
# PIPE_KWARGS = {"language": "vi", "task": "transcribe"}
BATCH_SIZE = 16


def evaluate(dataset):
    """
    Evaluate the model on the dataset and compute WER.

    Args:
        dataset: Hugging Face dataset object with audio and text columns.

    Returns:
        list: List of results containing reference, hypothesis, WER, and dataset attributes.
    """
    logging.info("Starting evaluation...")
    results = []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Evaluating"):
        batch = dataset[i : i + BATCH_SIZE]
        logging.info(f"Batch {i // BATCH_SIZE + 1} structure: {batch}")
        try:
            inputs = [audio["array"] for audio in batch["audio"]]
            outputs = PIPE(inputs)
            logging.info(
                f"Batch {i // BATCH_SIZE + 1} output type: {type(outputs)}, sample: {outputs[:2]}"
            )

            for j, output in enumerate(outputs):
                ref_text = batch["text"][j]
                hyp_text = output if isinstance(output, str) else output.get("text", "")
                logging.info(
                    f"Sample {i + j}: Reference: {ref_text}, Hypothesis: {hyp_text}"
                )
                if not hyp_text:
                    logging.warning(
                        f"Empty hypothesis for sample {i + j} (filename: {batch['filename'][j]}), skipping."
                    )
                    continue
                try:
                    wer = jiwer.wer(
                        ref_text,
                        hyp_text,
                        reference_transform=JIWER_TRANS,
                        hypothesis_transform=JIWER_TRANS,
                    )
                    results.append(
                        {
                            "reference": ref_text,
                            "hypothesis": hyp_text,
                            "wer": wer,
                            "filename": batch["filename"][j],
                            "province_code": batch["province_code"][j],
                            "province_name": batch["province_name"][j],
                            "speakerID": batch["speakerID"][j],
                            "gender": batch["gender"][j],
                        }
                    )
                    logging.info(
                        f"Sample {i + j}: WER = {wer:.4f}, "
                        f"Reference: {ref_text}, Hypothesis: {hyp_text}, "
                        f"Filename: {batch['filename'][j]}, Province: {batch['province_name'][j]}, "
                        f"SpeakerID: {batch['speakerID'][j]}, Gender: {'Male' if batch['gender'][j] == 1 else 'Female'}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error computing WER for sample {i + j} (filename: {batch['filename'][j]}): {str(e)}"
                    )
                    continue
        except Exception as e:
            logging.error(f"Error processing batch {i // BATCH_SIZE + 1}: {str(e)}")
            continue

    logging.info("Evaluation completed.")
    return results


def save_results(results, output_file="eval_results.txt"):
    """
    Save evaluation results to a file and compute overall WER and statistics by province and gender.

    Args:
        results (list): List of evaluation results.
        output_file (str): Path to save the results.
    """
    if not results:
        logging.warning("No results to save.")
        return

    wer = sum(result["wer"] for result in results) / len(results)
    wer_std = np.std([result["wer"] for result in results]) if results else 0.0
    logging.info(f"Overall WER: {wer:.4f} (Std: {wer_std:.4f})")

    province_wers = {}
    gender_wers = {"Male": [], "Female": []}
    for result in results:
        province = result["province_name"]
        if province not in province_wers:
            province_wers[province] = []
        province_wers[province].append(result["wer"])
        gender_key = "Male" if result["gender"] == 1 else "Female"
        gender_wers[gender_key].append(result["wer"])

    logging.info("WER by Province:")
    for province, wers in province_wers.items():
        mean_wer = sum(wers) / len(wers)
        logging.info(f"  {province}: {mean_wer:.4f} (n={len(wers)})")

    logging.info("WER by Gender:")
    for gender, wers in gender_wers.items():
        mean_wer = sum(wers) / len(wers) if wers else 0.0
        logging.info(f"  {gender}: {mean_wer:.4f} (n={len(wers)})")

    logging.info("Saving results to disk...")
    try:
        with open(os.path.join(log_dir, output_file), "w", encoding="utf-8") as f:
            for result in results:
                f.write(f"Filename: {result['filename']}\n")
                f.write(f"Province Code: {result['province_code']}\n")
                f.write(f"Province Name: {result['province_name']}\n")
                f.write(f"SpeakerID: {result['speakerID']}\n")
                f.write(f"Gender: {'Male' if result['gender'] == 1 else 'Female'}\n")
                f.write(f"Reference: {result['reference']}\n")
                f.write(f"Hypothesis: {result['hypothesis']}\n")
                f.write(f"WER: {result['wer']:.4f}\n\n")
            f.write(f"Overall WER: {wer:.4f} (Std: {wer_std:.4f})\n")
            f.write("WER by Province:\n")
            for province, wers in province_wers.items():
                mean_wer = sum(wers) / len(wers)
                f.write(f"  {province}: {mean_wer:.4f} (n={len(wers)})\n")
            f.write("WER by Gender:\n")
            for gender, wers in gender_wers.items():
                mean_wer = sum(wers) / len(wers) if wers else 0.0
                f.write(f"  {gender}: {mean_wer:.4f} (n={len(wers)})\n")
        logging.info("Results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        dataset = load_dataset(region="All")

    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

    results = evaluate(dataset)

    save_results(results, output_file="eval_results_all.txt")

    logging.info("Evaluation script completed successfully.")
    print("Evaluation completed. Check logs and results in the logs directory.")
