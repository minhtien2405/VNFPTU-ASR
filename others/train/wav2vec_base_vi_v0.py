import os
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_dataset, Audio
import evaluate
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from transformers import TrainerCallback
import numpy as np
import json
import logging

log_dir = os.path.join(os.getcwd(), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "wav2vec_base_vi_v0.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

load_dotenv("./configs/.env")
login(token=os.getenv("HF_TOKEN"))
mlflow.set_experiment("Wav2Vec2_Central_ViMD_FPTU")
cache_dir = os.getcwd() + "/cache"
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

dataset = load_dataset("nguyendv02/ViMD_Dataset", cache_dir=cache_dir)
train_dataset = dataset["train"].filter(lambda x: x["region"] == "Central")
valid_dataset = dataset["valid"].filter(lambda x: x["region"] == "Central")

train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(valid_dataset)}")
logging.info(
    f"Sample audio: {train_dataset[0]['audio']['array'][:5]}... (first 5 samples)"
)
logging.info(f"Data attributes: {train_dataset.features}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
processor = Wav2Vec2Processor.from_pretrained(model_id)


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


train_dataset = train_dataset.map(
    prepare_dataset, remove_columns=train_dataset.column_names
)
valid_dataset = valid_dataset.map(
    prepare_dataset, remove_columns=valid_dataset.column_names
)

logging.info(f"Processed train dataset: {train_dataset[0]}")
logging.info(f"Processed validation dataset: {valid_dataset[0]}")

model = Wav2Vec2ForCTC.from_pretrained(
    model_id,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

logging.info(f"Model loaded: {model_id}")


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


class MLflowCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_wer" in metrics:
            mlflow.log_metric("eval_wer", metrics["eval_wer"], step=state.global_step)


training_args = TrainingArguments(
    output_dir="./logs/wav2vec2-base-central-vi",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=100,
    save_total_limit=3,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    optim="adamw_torch",
    per_device_eval_batch_size=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="minhtien2405/wav2vec2-base-central-vi",
)
with mlflow.start_run(run_name="wav2vec2_finetune_central_vi"):
    mlflow.log_params(
        {
            "model_name": model_id,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "max_steps": training_args.max_steps,
            "warmup_steps": training_args.warmup_steps,
            "fp16": training_args.fp16,
        }
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback()],
    )

    trainer.train()
    logging.info("Training completed.")

    eval_results = trainer.evaluate()
    mlflow.log_metric("final_eval_wer", eval_results["eval_wer"])

    wer_history = [
        (log["step"], log["eval_wer"])
        for log in trainer.state.log_history
        if "eval_wer" in log
    ]
    with open("./logs/wav2vec2-base-central-vi/wer_history.json", "w") as f:
        json.dump(wer_history, f)
    mlflow.log_artifact(
        "./logs/wav2vec2-base-central-vi/wer_history.json", artifact_path="wer_history"
    )
    logging.info(f"Final evaluation results: {eval_results}")

    trainer.save_model("./models/wav2vec2-base-central-vi")
    processor.save_pretrained("./models/wav2vec2-base-central-vi")
    logging.info("Model and processor saved locally.")

    # mlflow.pytorch.log_model(model, "model")
    # mlflow.log_artifact("./wav2vec2-base-central-vi", artifact_path="processor")

    trainer.push_to_hub(
        commit_message="Fine-tuned Wav2Vec2 on ViMD Central region",
        tags=["speech-recognition", "vietnamese", "central-vietnam"],
        dataset="nguyendv02/ViMD_Dataset",
        language="vi",
        finetuned_from=model_id,
        tasks="automatic-speech-recognition",
    )
    processor.push_to_hub("minhtien2405/wav2vec2-base-central-vi")
    logging.info("Model and processor pushed to Hugging Face Hub.")

# torchrun --nproc_per_node=2 wav2vec_base_vi_v0.py
