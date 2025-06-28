# PhoWhisper Speech-to-Text

A professional setup for fine-tuning, evaluating, and running inference with the PhoWhisper model for Vietnamese speech-to-text tasks, supporting region-specific training and evaluation (All, Central, South, North).

## Setup

1. Set up environment variables:
   Create a `.env` file in `phowhisper/train/configs/` with:
   ```plaintext
   HF_TOKEN=your_huggingface_token
   WANDB_API_KEY=your_wandb_api_key
   ```

2. Configure Weights & Biases:
   Log in to WandB:
   ```bash
   wandb login
   ```

3. convert phowhisper model to transformers format:
   ```bash
   ct2-transformers-converter \
  --model vinai/PhoWhisper-large \
  --output_dir ./converted_phowhisper \
  --copy_files tokenizer.json tokenizer_config.json special_tokens_map.json \
  --quantization float16
   ```

## Usage

### Training
Run training for a specific region (e.g., Central):
```bash
bash ASR/Phowhisper/scripts/train.sh Central
```

### Evaluation
Evaluate the model for the same region used in training:
```bash
bash ASR/Phowhisper/scripts/eval.sh --model model_name
```

### Inference
Run inference on an audio file for a specific region:
```bash
bash ASR/Phowhisper/scripts/infer.sh path/to/audio.wav --model model_name
```

## Configuration
Edit `phowhisper/configs/config.yaml` to adjust model, dataset, and training parameters. The `regions` field lists supported regions: `All`, `Central`, `South`, `North`.

## Evaluation Results
- Results are saved to `logs/phowhisper-large-{region}-vi/eval_results.json`.
- Metrics include:
  - Total WER
  - Average inference latency (per file)
  - Total latency
  - WER by Province
  - WER by Gender
  - Per-file results (Filename, Province Code, Province Name, SpeakerID, Gender, Reference, Hypothesis, WER)
- Results are also logged to Weights & Biases under the project `PhoWhisper_ViMD`.

## Directory Structure
- `train/`: Training scripts, configuration, and `.env` file.
- `eval/`: Evaluation scripts.
- `infer/`: Inference scripts.
- `scripts/`: Shell scripts for running tasks.
- `cli.py`: CLI interface using Click.