import os
import re
import logging
import requests
import torch
import jiwer
import numpy as np
import datasets as hugDS
import librosa
from tqdm import tqdm
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)
from peft import PeftModel, PeftConfig
from huggingface_hub import login
from dotenv import load_dotenv

# --- CONFIGURATION ---
PEFT_MODEL_ID = "minhtien2405/vovinam-phowhisper-large-vi"
DATASET_ID = "minhtien2405/VoviAIDataset"
DATASET_SPLIT = "test"
BATCH_SIZE = 8  # Giảm nếu gặp lỗi hết bộ nhớ (Out of Memory)
CACHE_DIR = os.path.join(os.getcwd(), "cache")
LOG_DIR = os.path.join(os.getcwd(), "logs")

# --- LOGGING SETUP ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

LOG_FILENAME = os.path.join(LOG_DIR, "eval_phowhisper_vovi.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# --- HUGGING FACE LOGIN ---
try:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.warning("Hugging Face token not found in .env file. Proceeding without login.")
    else:
        login(token=hf_token)
        logging.info("Hugging Face login successful.")
except Exception as e:
    logging.error(f"Could not log in to Hugging Face: {e}")


# --- TEXT NORMALIZATION & DATA UTILS ---
JIWER_TRANS = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def normalize_text(text: str) -> str:
    """A simple function to normalize text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def download_audio_from_s3(url: str, cache_dir: str) -> str | None:
    """Downloads an audio file from a URL and saves it to a cache directory."""
    if not url:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("?")[0].split("/")[-1]
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return None


def load_and_prepare_data(dataset_id: str, split: str, cache_dir: str):
    """Loads, downloads, and validates the VoviAI dataset."""
    logging.info(f"Loading dataset '{dataset_id}' on split '{split}'")
    dataset = hugDS.load_dataset(dataset_id, split=split, cache_dir=cache_dir)
    
    audio_cache_dir = os.path.join(cache_dir, "vovi_audio")

    def process_sample(sample):
        sample["is_valid"] = False
        local_audio_path = download_audio_from_s3(sample["audioLink"], cache_dir=audio_cache_dir)
        
        if not local_audio_path:
            return sample

        sample["local_audio_path"] = local_audio_path
        
        try:
            duration = librosa.get_duration(path=local_audio_path)
            if not (0.1 < duration < 30):
                logging.warning(f"Skipping {local_audio_path} due to invalid duration: {duration:.2f}s")
                return sample
        except Exception as e:
            logging.error(f"Could not read duration for {local_audio_path}: {e}")
            return sample
        
        text = normalize_text(sample.get("text", ""))
        if len(text) < 2:
            logging.warning(f"Skipping {local_audio_path} due to short transcript: '{text}'")
            return sample
            
        sample["text"] = text
        sample["is_valid"] = True
        return sample
    
    logging.info("Downloading and validating audio files...")
    dataset = dataset.map(process_sample, num_proc=4)
    
    initial_count = len(dataset)
    dataset = dataset.filter(lambda s: s["is_valid"], num_proc=4)
    final_count = len(dataset)
    
    logging.info(f"Finished validation. Kept {final_count}/{initial_count} samples.")
    return dataset


def evaluate(dataset, pipeline):
    """Evaluates the model on the dataset and computes WER."""
    logging.info("Starting evaluation...")
    results = []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Evaluating"):
        batch = dataset[i : i + BATCH_SIZE]
        
        try:
            # The pipeline can directly process a list of local file paths
            inputs = batch["local_audio_path"]
            outputs = pipeline(inputs, batch_size=len(inputs))

            for j, output in enumerate(outputs):
                ref_text = batch["text"][j]
                hyp_text = output.get("text", "").lower()
                
                if not hyp_text:
                    logging.warning(f"Empty hypothesis for sample {i+j}, skipping.")
                    continue
                
                wer = jiwer.wer(
                    ref_text,
                    hyp_text,
                    reference_transform=JIWER_TRANS,
                    hypothesis_transform=JIWER_TRANS,
                )
                
                results.append({
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "wer": wer,
                    "audioLink": batch["audioLink"][j],
                })
        except Exception as e:
            logging.error(f"Error processing batch starting at index {i}: {e}", exc_info=True)
            continue

    logging.info("Evaluation completed.")
    return results


def save_results(results, output_file):
    """Saves evaluation results to a file."""
    if not results:
        logging.warning("No results to save.")
        return

    all_wers = [res["wer"] for res in results]
    overall_wer = sum(all_wers) / len(all_wers) if all_wers else 0
    wer_std = np.std(all_wers) if all_wers else 0

    logging.info(f"Overall WER: {overall_wer:.4f} (Std: {wer_std:.4f})")
    
    logging.info(f"Saving detailed results to: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {PEFT_MODEL_ID}\n")
            f.write(f"DATASET: {DATASET_ID} ({DATASET_SPLIT})\n\n")
            f.write("--- OVERALL RESULTS ---\n")
            f.write(f"Word Error Rate (WER): {overall_wer:.4f} ({overall_wer*100:.2f}%)\n")
            f.write(f"Standard Deviation: {wer_std:.4f}\n")
            f.write(f"Total Samples: {len(results)}\n\n")
            f.write("="*50 + "\n\n")
            f.write("--- DETAILED RESULTS PER SAMPLE ---\n\n")
            
            for res in results:
                f.write(f"Audio Link: {res['audioLink']}\n")
                f.write(f"Reference:  {res['reference']}\n")
                f.write(f"Hypothesis: {res['hypothesis']}\n")
                f.write(f"WER: {res['wer']:.4f}\n\n")
        logging.info("Results saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save results file: {e}")


def main():
    """Main function to run the evaluation script."""
    # --- Load Data ---
    dataset = load_and_prepare_data(DATASET_ID, DATASET_SPLIT, CACHE_DIR)
    
    # --- Load Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logging.info(f"Using device: {device} with dtype: {torch_dtype}")

    try:
        # Check if it's a PEFT model to load the base model first
        try:
            config = PeftConfig.from_pretrained(PEFT_MODEL_ID, cache_dir=CACHE_DIR)
            base_model_id = config.base_model_name_or_path
            logging.info(f"Loading base model '{base_model_id}' with PEFT adapter '{PEFT_MODEL_ID}'")
        except Exception:
            # If not a PEFT model, treat the ID as the final model
            base_model_id = PEFT_MODEL_ID
            logging.info(f"Loading full model (not PEFT): '{base_model_id}'")

        feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_id, cache_dir=CACHE_DIR)
        tokenizer = WhisperTokenizer.from_pretrained(base_model_id, language="vi", task="transcribe", cache_dir=CACHE_DIR)
        
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            cache_dir=CACHE_DIR,
        )

        if base_model_id != PEFT_MODEL_ID:
            model = PeftModel.from_pretrained(model, PEFT_MODEL_ID, cache_dir=CACHE_DIR)
            model = model.merge_and_unload()

        model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="vi", task="transcribe")
        model.to(device)

        # --- Create Pipeline ---
        pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=0 if device.type == "cuda" else -1, # Use device index for pipeline
            torch_dtype=torch_dtype,
        )
        logging.info("ASR Pipeline created successfully.")

    except Exception as e:
        logging.error(f"Failed to load model or create pipeline: {e}", exc_info=True)
        return

    # --- Evaluate and Save ---
    results = evaluate(dataset, pipeline)
    output_filepath = os.path.join(LOG_DIR, "results_phowhisper_vovi.txt")
    save_results(results, output_filepath)

    logging.info("Evaluation script finished successfully.")
    print("\n✅ Evaluation complete!")
    print(f"Check logs at: {LOG_FILENAME}")
    print(f"Check detailed results at: {output_filepath}")


if __name__ == "__main__":
    main()