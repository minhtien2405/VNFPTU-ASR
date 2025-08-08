import os
import re
import logging
import torch
import jiwer
import numpy as np
import datasets as hugDS
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- CẤU HÌNH ---
MODEL_ID = "minhtien2405/wav2vec2-base-vi"
DATASET_ID = "nguyendv02/ViMD_Dataset"
DATASET_SPLIT = "test"
BATCH_SIZE = 8  # Giảm nếu gặp lỗi hết bộ nhớ (Out of Memory)
CACHE_DIR = os.path.join(os.getcwd(), "cache")
LOG_DIR = os.path.join(os.getcwd(), "logs")

# --- THIẾT LẬP LOGGING ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

LOG_FILENAME = os.path.join(LOG_DIR, "eval_wav2vec2_vimd.log")
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# --- BỘ CHUYỂN ĐỔI VĂN BẢN JIWER ---
# Dùng để chuẩn hóa văn bản tham chiếu và văn bản dự đoán trước khi tính WER
JIWER_TRANS = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def normalize_text(text):
    """Hàm chuẩn hóa văn bản đơn giản."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Xóa các ký tự không phải chữ cái, số, hoặc khoảng trắng
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(dataset_id, split):
    """Tải và tiền xử lý dataset ViMD."""
    logging.info(f"Đang tải dataset: {dataset_id}, split: {split}")
    try:
        dataset = hugDS.load_dataset(dataset_id, split=split, cache_dir=CACHE_DIR)
        # Đảm bảo tần số lấy mẫu là 16kHz
        dataset = dataset.cast_column("audio", hugDS.Audio(sampling_rate=16000))
        logging.info(f"Tải dataset thành công. Số lượng mẫu ban đầu: {len(dataset)}")
    except Exception as e:
        logging.error(f"Lỗi khi tải dataset: {e}")
        raise

    # Lọc các mẫu có độ dài audio > 30 giây để tránh lỗi OOM
    initial_count = len(dataset)
    dataset = dataset.filter(lambda x: x["audio"] is not None and len(x["audio"]["array"]) <= 30 * 16000, num_proc=4)
    filtered_count = len(dataset)
    logging.info(f"Đã lọc các mẫu > 30s hoặc audio rỗng. Giữ lại {filtered_count}/{initial_count} mẫu.")
    
    # Chuẩn hóa cột text
    dataset = dataset.map(lambda x: {"text": normalize_text(x["text"])}, num_proc=4)
    logging.info("Đã chuẩn hóa cột 'text'.")

    return dataset

def evaluate(dataset, model, processor, device):
    """
    Đánh giá mô hình trên dataset và tính toán WER.
    """
    logging.info("Bắt đầu quá trình đánh giá...")
    results = []

    # Vòng lặp qua dataset theo từng batch
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Đang đánh giá"):
        batch = dataset[i : i + BATCH_SIZE]
        
        try:
            # Chuẩn bị audio input
            # Dòng này đã được sửa lại cho đúng
            audio_inputs = [audio_sample["array"] for audio_sample in batch["audio"]]
            
            inputs = processor(
                audio_inputs, sampling_rate=16000, return_tensors="pt", padding=True
            )

            # Đưa input lên device (GPU/CPU)
            input_values = inputs.input_values.to(device)
            
            # Chạy suy luận
            with torch.no_grad():
                logits = model(input_values).logits

            # Giải mã logits thành ID và sau đó thành văn bản
            predicted_ids = torch.argmax(logits, dim=-1)
            hypotheses = processor.batch_decode(predicted_ids)
            
            references = batch["text"]

            # Xử lý kết quả cho từng mẫu trong batch
            for j in range(len(references)):
                ref_text = references[j]
                hyp_text = hypotheses[j].lower() # Chuyển kết quả dự đoán về chữ thường

                # Tính WER
                wer = jiwer.wer(
                    ref_text,
                    hyp_text,
                    reference_transform=JIWER_TRANS,
                    hypothesis_transform=JIWER_TRANS,
                )
                
                # Lưu kết quả
                results.append({
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "wer": wer,
                    "filename": batch["filename"][j],
                    "province_name": batch["province_name"][j],
                    "gender": batch["gender"][j],
                })
        
        except Exception as e:
            logging.error(f"Lỗi khi xử lý batch bắt đầu từ vị trí {i}: {e}")
            continue
            
    logging.info("Đánh giá hoàn tất.")
    return results

def save_results(results, output_file):
    """Lưu kết quả đánh giá và tính toán các chỉ số thống kê."""
    if not results:
        logging.warning("Không có kết quả để lưu.")
        return

    # Tính WER tổng thể và độ lệch chuẩn
    all_wers = [res["wer"] for res in results]
    overall_wer = sum(all_wers) / len(all_wers)
    wer_std = np.std(all_wers)

    logging.info(f"WER Tổng thể: {overall_wer:.4f} (Std: {wer_std:.4f})")

    # Phân tích WER theo tỉnh thành
    province_wers = {}
    for res in results:
        province = res["province_name"]
        if province not in province_wers:
            province_wers[province] = []
        province_wers[province].append(res["wer"])

    # Phân tích WER theo giới tính
    gender_wers = {"Male": [], "Female": []}
    for res in results:
        gender_key = "Male" if res["gender"] == 1 else "Female"
        gender_wers[gender_key].append(res["wer"])

    # Ghi log các kết quả thống kê
    logging.info("--- WER theo Tỉnh thành ---")
    for province, wers in sorted(province_wers.items()):
        mean_wer = sum(wers) / len(wers)
        logging.info(f"  {province}: {mean_wer:.4f} (n={len(wers)})")

    logging.info("--- WER theo Giới tính ---")
    for gender, wers in gender_wers.items():
        mean_wer = sum(wers) / len(wers) if wers else 0.0
        logging.info(f"  {gender}: {mean_wer:.4f} (n={len(wers)})")
        
    # Lưu kết quả chi tiết ra file
    logging.info(f"Đang lưu kết quả chi tiết vào: {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {MODEL_ID}\n")
            f.write(f"DATASET: {DATASET_ID} ({DATASET_SPLIT})\n\n")
            f.write("--- KẾT QUẢ TỔNG THỂ ---\n")
            f.write(f"Overall WER: {overall_wer:.4f} (Std: {wer_std:.4f})\n\n")

            f.write("--- THỐNG KÊ WER THEO TỈNH THÀNH ---\n")
            for province, wers in sorted(province_wers.items()):
                mean_wer = sum(wers) / len(wers)
                f.write(f"  {province}: {mean_wer:.4f} (n={len(wers)})\n")
            f.write("\n")

            f.write("--- THỐNG KÊ WER THEO GIỚI TÍNH ---\n")
            for gender, wers in gender_wers.items():
                mean_wer = sum(wers) / len(wers) if wers else 0.0
                f.write(f"  {gender}: {mean_wer:.4f} (n={len(wers)})\n")
            f.write("\n" + "="*50 + "\n\n")

            f.write("--- KẾT QUẢ CHI TIẾT TỪNG MẪU ---\n")
            for res in results:
                f.write(f"Filename: {res['filename']}\n")
                f.write(f"Province: {res['province_name']}\n")
                f.write(f"Gender: {'Male' if res['gender'] == 1 else 'Female'}\n")
                f.write(f"Reference:  {res['reference']}\n")
                f.write(f"Hypothesis: {res['hypothesis']}\n")
                f.write(f"WER: {res['wer']:.4f}\n\n")
        logging.info("Lưu file kết quả thành công.")
    except Exception as e:
        logging.error(f"Lỗi khi lưu file kết quả: {e}")

def main():
    """Hàm chính điều khiển luồng thực thi."""
    logging.info("Bắt đầu script đánh giá Wav2Vec2.")
    
    # Chọn device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Sử dụng device: {device}")

    # Tải processor và model
    try:
        logging.info(f"Đang tải processor từ: {MODEL_ID}")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        
        logging.info(f"Đang tải model từ: {MODEL_ID}")
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        model.to(device)
        model.eval() # Chuyển model sang chế độ đánh giá
    except Exception as e:
        logging.error(f"Không thể tải model hoặc processor: {e}")
        return

    # Tải và chuẩn bị dữ liệu
    dataset = load_data(DATASET_ID, DATASET_SPLIT)
    
    # Thực hiện đánh giá
    evaluation_results = evaluate(dataset, model, processor, device)
    
    # Lưu kết quả
    output_filename = os.path.join(LOG_DIR, "results_wav2vec2_vimd_test.txt")
    save_results(evaluation_results, output_filename)

    logging.info("Script đánh giá đã hoàn thành.")
    print("\n✅ Đánh giá hoàn tất!")
    print(f"Kiểm tra file log tại: {LOG_FILENAME}")
    print(f"Kiểm tra file kết quả tại: {output_filename}")

if __name__ == "__main__":
    main()