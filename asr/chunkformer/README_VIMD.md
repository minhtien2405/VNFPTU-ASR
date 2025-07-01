# Chunkformer Training on VIMD Dataset

Hướng dẫn thiết lập và training Chunkformer với dataset VIMD (Vietnamese Multi-Domain) cho task ASR.

## 📋 Tổng quan

Setup này cho phép bạn finetune mô hình Chunkformer trên dataset VIMD với cấu trúc tương tự như PhoWhisper, nhưng được tối ưu cho khả năng xử lý audio dài của Chunkformer.

### Tính năng chính:
- **Long-form Audio Processing**: Xử lý audio dài đến 16 tiếng
- **Efficient Memory Usage**: Tối ưu bộ nhớ với Masked Batch technique
- **Multi-region Training**: Hỗ trợ training theo vùng miền (All, Central, South, North)
- **WeNet Integration**: Tích hợp với framework WeNet cho ASR
- **WandB Logging**: Theo dõi training với Weights & Biases

## 🔧 Cài đặt

### 1. Cài đặt Dependencies

```bash
# Cài đặt các package cơ bản
pip install -r requirements_vimd.txt

# Cài đặt WeNet (bắt buộc)
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
pip install -e .
cd ..
```

### 2. Cấu hình Environment

```bash
# Thiết lập environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Hoặc GPU IDs bạn muốn sử dụng

# Đăng nhập WandB (optional)
wandb login
```

## 📊 Dataset

Dataset VIMD (`nguyendv02/ViMD_Dataset`) được tự động tải từ HuggingFace Hub. Dataset bao gồm:

- **Train/Valid splits** cho mỗi vùng miền
- **4 regions**: All, Central, South, North
- **Audio format**: WAV files, 16kHz sampling rate
- **Text format**: Vietnamese transcriptions

## ⚙️ Cấu hình

### File cấu hình chính: `configs/config_vimd.yaml`

```yaml
model:
  model_id: "khanhld/chunkformer-large-vie"  # Base checkpoint
  chunk_size: 64
  left_context_size: 128
  right_context_size: 128

dataset:
  name: "nguyendv02/ViMD_Dataset"
  sampling_rate: 16000
  regions: ["All", "Central", "South", "North"]

training:
  learning_rate: 2e-5
  num_train_epochs: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
```

### Tùy chỉnh cấu hình:

- **Chunk parameters**: Điều chỉnh `chunk_size`, `left_context_size`, `right_context_size`
- **Training parameters**: Learning rate, batch size, epochs
- **Data processing**: Audio chunking, text normalization settings

## 🚀 Training

### 1. Training đơn region

```bash
# Training cho region Central
python train_chunkformer.py \
    --config configs/config_vimd.yaml \
    --region Central \
    --log-level INFO

# Training với custom settings
python train_chunkformer.py \
    --config configs/config_vimd.yaml \
    --region All \
    --device cuda \
    --log-level DEBUG
```

### 2. Training tất cả regions

```bash
# Sử dụng script tự động
bash scripts/train_all_regions.sh

# Với options
bash scripts/train_all_regions.sh --log-level DEBUG --continue-on-error
```

### 3. Dry run (kiểm tra setup)

```bash
# Test data preparation
python train_chunkformer.py \
    --config configs/config_vimd.yaml \
    --region Central \
    --dry-run

# Test tất cả regions
bash scripts/train_all_regions.sh --dry-run
```

## 📈 Monitoring

### WandB Integration

Training metrics được tự động ghi vào WandB:
- Training/validation loss
- WER (Word Error Rate)
- CER (Character Error Rate)
- Dataset statistics
- Model parameters

### Local Logs

Logs được lưu trong:
```
logs/
├── chunkformer_vimd_all_v1.log
├── chunkformer_vimd_central_v1.log
├── chunkformer_vimd_south_v1.log
└── chunkformer_vimd_north_v1.log
```

## 🎯 Evaluation

### Tự động evaluation

```bash
# Evaluation được chạy tự động sau training
python train_chunkformer.py --config configs/config_vimd.yaml --region Central
```

### Manual evaluation

```bash
# Chỉ chạy evaluation
python train_chunkformer.py \
    --config configs/config_vimd.yaml \
    --region Central \
    --eval-only
```

## 📁 Cấu trúc Output

```
logs/chunkformer-vimd-{region}-vi/
├── wenet_config.yaml          # WeNet configuration
├── vocab.txt                  # Vocabulary file
├── data/                      # Processed data
│   ├── train_data.list       # Training data list
│   ├── valid_data.list       # Validation data list
│   └── audio_chunks/         # Chunked audio files
├── checkpoint/               # Training checkpoints
├── final.pt                  # Final model
└── best.pt                   # Best model

outputs/chunkformer-vimd-{region}-vi/
├── recognition_results.txt   # Recognition results
├── metrics.json             # Evaluation metrics
└── eval_results.json        # Detailed evaluation
```

## 🔧 Troubleshooting

### Common Issues

1. **WeNet not found**:
   ```bash
   # Reinstall WeNet
   pip uninstall wenet
   git clone https://github.com/wenet-e2e/wenet.git
   cd wenet && pip install -e .
   ```

2. **CUDA out of memory**:
   ```yaml
   # Reduce batch size in config
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 8
   ```

3. **Dataset loading issues**:
   ```bash
   # Clear cache and retry
   rm -rf ./cache
   python train_chunkformer.py --config configs/config_vimd.yaml --region Central --dry-run
   ```

### Debug Mode

```bash
# Enable detailed logging
python train_chunkformer.py \
    --config configs/config_vimd.yaml \
    --region Central \
    --log-level DEBUG
```

## 📊 Performance Notes

### Memory Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **For long audio (>1 hour)**: 24GB+ recommended

### Training Time

Estimated training time per region:
- **Central/South/North**: 2-4 hours (depending on GPU)
- **All regions**: 6-8 hours

### Batch Duration Settings

```yaml
model:
  total_batch_duration: 14400  # 4 hours worth of audio
  
# Adjust based on GPU memory:
# - 8GB GPU: 3600 (1 hour)
# - 16GB GPU: 7200 (2 hours) 
# - 24GB GPU: 14400 (4 hours)
```

## 🎯 Advanced Usage

### Custom Data Processing

```python
# Modify processor/data_processor.py
class ChunkformerDataProcessor:
    def __init__(self, config, device="cuda"):
        # Custom chunker settings
        self.chunker = AudioChunker(
            chunk_overlap=0.15,  # 15% overlap
            max_chunk_length=45.0,  # 45 seconds max
            # ... other settings
        )
```

### Model Customization

```yaml
# In config_vimd.yaml
model:
  # Chunkformer specific parameters
  chunk_size: 128        # Larger chunks
  left_context_size: 256 # More context
  right_context_size: 256
```

### Distributed Training

```bash
# Multi-GPU training (experimental)
torchrun --nproc_per_node=2 train_chunkformer.py \
    --config configs/config_vimd.yaml \
    --region All
```

## 📚 References

- [Chunkformer Paper](docs/paper.pdf)
- [WeNet Documentation](https://github.com/wenet-e2e/wenet)
- [VIMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset)
- [Original Chunkformer Repository](../README.md)

## 🤝 Contributing

Contributions are welcome! Please see the main project's contributing guidelines.

## 📄 License

This project follows the same license as the main Chunkformer repository.

---

**Note**: Đây là phiên bản beta. Vui lòng báo cáo issues nếu gặp vấn đề trong quá trình training.