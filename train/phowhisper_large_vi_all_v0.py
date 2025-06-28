import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
import evaluate
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
import json
import logging
import peft
import accelerate

log_dir = os.getcwd() + "/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, "phowhisper_large_vi_all_v0.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
load_dotenv("./configs/.env")
login(token=os.getenv("HF_TOKEN"))

mlflow.set_experiment("PhoWhisper_ViMD_FPTU")

cache_dir = os.getcwd() + "/cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

dataset = load_dataset("nguyendv02/ViMD_Dataset", cache_dir=cache_dir)

train_dataset = dataset["train"].cast_column("audio", Audio(sampling_rate=16000))
valid_dataset = dataset["valid"].cast_column("audio", Audio(sampling_rate=16000))

num_of_long_audio = 0


def filter_long_audio(example):
    global num_of_long_audio
    audio_length = (
        example["audio"]["array"].shape[0] / example["audio"]["sampling_rate"]
    )
    if audio_length > 30:
        num_of_long_audio += 1
        logging.warning(
            f"Audio {example['audio']['path']} is too long: {audio_length:.2f} seconds"
        )
    return audio_length <= 30


logging.info(f"Number of long audio samples: {num_of_long_audio}")

train_dataset = train_dataset.filter(filter_long_audio)
valid_dataset = valid_dataset.filter(filter_long_audio)

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(valid_dataset)}")
logging.info(
    f"Sample audio: {train_dataset[0]['audio']['array'][:5]}... (first 5 samples)"
)
logging.info(f"Data attributes: {train_dataset.features}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model_id = "vinai/PhoWhisper-large"
processor = WhisperProcessor.from_pretrained(model_id, language="vi", task="transcribe")


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
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

model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    use_cache=False,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="vi", task="transcribe"
)
model.config.suppress_tokens = []

if torch.cuda.device_count() > 1:
    DEV_MAP = model.hf_device_map.copy()
    DEV_MAP["model.decoder.embed_tokens"] = DEV_MAP[
        "model.decoder.embed_positions"
    ] = DEV_MAP["proj_out"] = model._hf_hook.execution_device
    accelerate.dispatch_model(model, device_map=DEV_MAP)
    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

logging.info(f"Model loaded: {model_id}")

peft_model = peft.get_peft_model(
    peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    ),
    peft.LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    ),
    # peft.AdaLoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=.05, bias="none")
)

peft_model.model.model.encoder.conv1.register_forward_hook(
    lambda module, input, output: output.requires_grad_(True)
)  # re-enable grad computation for conv layer
peft_model.print_trainable_parameters()  # 16 millions = 1% of 1.6 billions params of whisper large


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


class MLflowCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_wer" in metrics:
            mlflow.log_metric("eval_wer", metrics["eval_wer"], step=state.global_step)


training_args = Seq2SeqTrainingArguments(
    output_dir="./logs/phowhisper-large-all-vi",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=100,
    # max_steps=1000, # testing
    gradient_checkpointing=True,
    fp16=True,
    optim="adamw_bnb_8bit",
    eval_strategy="no",
    per_device_eval_batch_size=4,
    save_steps=100,
    # eval_steps=100,
    save_total_limit=3,
    logging_steps=50,
    # load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    remove_unused_columns=False,
    hub_model_id="minhtien2405/phowhisper-large-all-vi",
)

with mlflow.start_run(run_name="phowhisper_finetune_all_vi"):
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

    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MLflowCallback()],
    )

    trainer.train()  # resume_from_checkpoint=True)
    logging.info("Training completed.")

    try:
        eval_results = trainer.evaluate()
        # mlflow.log_metric("final_eval_wer", eval_results["eval_wer"])
        logging.info(
            f"Final evaluation WER in validation set: {eval_results['eval_wer']}"
        )

        with open("./logs/phowhisper-large-all-vi/eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)
        logging.info("Evaluation results saved to eval_results.json.")
        mlflow.log_artifact(
            "./logs/phowhisper-large-all-vi/eval_results.json",
            artifact_path="eval_results",
        )

    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error("Out of memory error during evaluation. Skipping evaluation.")
        else:
            logging.error(f"Runtime error during evaluation: {e}")
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

    model_dir = "./models/phowhisper-large-all-vi"
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    logging.info("Model and processor saved locally.")

    trainer.push_to_hub(
        commit_message="Fine-tuned PhoWhisper large on ViMD All region",
        tags=["speech-recognition", "vietnamese", "all-vietnam"],
        dataset="nguyendv02/ViMD_Dataset",
        language="vi",
        finetuned_from=model_id,
        tasks="automatic-speech-recognition",
    )
    processor.push_to_hub("minhtien2405/phowhisper-large-all-vi")
    logging.info("Model pushed to Hugging Face Hub.")
