import logging
import torch
import time
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from datasets import load_dataset, Audio
import wandb
import json
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config, model_path: str):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained(
            config.model.model_id, language=config.model.language, task=config.model.task
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        wandb.init(project=config.wandb.project, name=f"eval_phowhisper_{config.region.lower()}")
        logger.info(f"Loaded model from {model_path} for evaluation")

    def load_dataset(self):
        dataset = load_dataset(self.config.dataset.name, cache_dir=self.config.dataset.cache_dir)
        valid_dataset = dataset["valid"].cast_column("audio", Audio(sampling_rate=self.config.dataset.sampling_rate))

        # Filter by region if not "All"
        if self.config.region != "All":
            valid_dataset = valid_dataset.filter(lambda x: x["region"] == self.config.region)
            logger.info(f"Filtered validation dataset for region {self.config.region}")
        logger.info(f"Loaded validation dataset: size={len(valid_dataset)}")
        return valid_dataset

    def evaluate(self):
        dataset = self.load_dataset()
        metric = evaluate.load("wer")
        total_latency = 0
        latencies = []
        per_file_results = []
        wer_by_province = defaultdict(list)
        wer_by_gender = defaultdict(list)

        def compute_metrics(batch):
            nonlocal total_latency
            start_time = time.time()
            audio = batch["audio"]
            input_features = self.processor(
                audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
            ).input_features.to(self.device)
            labels = self.processor.tokenizer(batch["text"]).input_ids
            with torch.no_grad():
                pred_ids = self.model.generate(input_features)
            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            label_str = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)[0]
            wer = 100 * metric.compute(predictions=[pred_str], references=[label_str])
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Collect per-file results
            result = {
                "Filename": batch["filename"],
                "Province Code": batch["province_code"],
                "Province Name": batch["province_name"],
                "SpeakerID": batch["speakerID"],
                "Gender": batch["gender"],
                "Reference": label_str,
                "Hypothesis": pred_str,
                "WER": wer
            }
            per_file_results.append(result)
            
            # Group WER by province and gender
            wer_by_province[batch["province_name"]].append(wer)
            wer_by_gender[batch["gender"]].append(wer)
            
            wandb.log({"eval_latency": latency})
            logger.info(f"Evaluated {result['Filename']}: WER={wer:.2f}, Latency={latency:.2f}s")
            return {"wer": wer}

        dataset.map(compute_metrics)
        total_wer = sum(per_file_results[i]["WER"] for i in range(len(per_file_results))) / len(per_file_results) if per_file_results else 0
        average_infer_latency = sum(latencies) / len(latencies) if latencies else 0
        total_latency = sum(latencies)

        # Calculate average WER by province and gender
        wer_by_province_avg = {prov: sum(wers) / len(wers) for prov, wers in wer_by_province.items()}
        wer_by_gender_avg = {gender: sum(wers) / len(wers) for gender, wers in wer_by_gender.items()}

        # Prepare results
        results = {
            "total_wer": total_wer,
            "average_infer_latency": average_infer_latency,
            "total_latency": total_latency,
            "wer_by_province": wer_by_province_avg,
            "wer_by_gender": wer_by_gender_avg,
            "per_file_results": per_file_results
        }

        # Save to JSON
        eval_path = os.path.join(self.config.training.output_dir, "eval_results.json")
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {eval_path}")

        # Log to WandB
        wandb.log({
            "total_wer": total_wer,
            "average_infer_latency": average_infer_latency,
            "total_latency": total_latency,
            "wer_by_province": wer_by_province_avg,
            "wer_by_gender": wer_by_gender_avg
        })
        wandb_table = wandb.Table(columns=[
            "Filename", "Province Code", "Province Name", "SpeakerID", "Gender", 
            "Reference", "Hypothesis", "WER"
        ])
        for result in per_file_results:
            wandb_table.add_data(
                result["Filename"], result["Province Code"], result["Province Name"],
                result["SpeakerID"], result["Gender"], result["Reference"],
                result["Hypothesis"], result["WER"]
            )
        wandb.log({"per_file_results": wandb_table})
        wandb.save(eval_path)
        wandb.finish()

        logger.info(f"Evaluation summary: Total WER={total_wer:.2f}, "
                    f"Avg Latency={average_infer_latency:.2f}s, Total Latency={total_latency:.2f}s")
        return results