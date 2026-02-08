# Sweet Potato Root Detection & Segmentation with YOLOv8

**Tuskegee AIFARMS Agricultural AI Research**

A complete, production-ready training pipeline for sweet potato root detection and instance segmentation using Ultralytics YOLOv8. This pipeline is designed for agricultural yield estimation and quality assessment, with real-time inference capabilities for field deployment.

## 🎯 Features

- **Root-Specific Preprocessing**: Handles soil backgrounds, overlapping roots, and lighting variations
- **Custom Metrics**: Root count per image, total root area coverage, size distribution analysis
- **Active Learning**: Flags low-confidence predictions for re-annotation
- **Model Comparison**: Baseline setup for YOLOv8 vs Mask R-CNN comparison
- **Transfer Learning**: Fine-tune from agricultural/plant detection checkpoints
- **Production Exports**: ONNX and TorchScript models for edge deployment
- **Comprehensive Evaluation**: Confusion matrices, PR curves, mAP scores, and visualizations

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (T4/V100 recommended for Colab)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB free space for dataset and models

### Software
- Google Colab (recommended) or local Python 3.8+ environment
- CUDA-capable PyTorch installation
- See `requirements.txt` for exact package versions

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload Dataset to Google Drive**
   - Place `Sweetpotato_roots.v2i.yolov8 (1).zip` in your Google Drive
   - Note the exact filename (update in notebook if different)

2. **Open Notebook**
   - Upload `sweetpotato_yolov8_training.ipynb` to Google Colab
   - Enable GPU: Runtime → Change runtime type → GPU (T4 or V100)

3. **Run All Cells**
   - Mount Google Drive when prompted
   - The notebook will automatically:
     - Install dependencies
     - Locate and extract dataset
     - Validate data structure
     - Train the model
     - Evaluate and export results

4. **Download Results**
   - Results are automatically packaged and downloaded
   - Includes: `best.pt`, `model.onnx`, `predictions.csv`, visualizations

### Option 2: Local Environment

```bash
# Clone or download this repository
cd farm-photo-outliner

# Install dependencies
pip install -r requirements.txt

# Update paths in the notebook or convert to Python script
# Run training pipeline
```

## 📁 Dataset Structure

The pipeline expects a YOLOv8-formatted dataset with the following structure:

```
Sweetpotato_roots.v2i.yolov8/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### data.yaml Format

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 2  # Number of classes
names: ['sweetpotato_root', 'background']
```

## ⚙️ Configuration

### Hyperparameters

Edit `config.yaml` to customize training:

```yaml
model: yolov8m-seg.pt  # Model size (n/s/m/l/x)
epochs: 100
batch: 16              # Reduce if GPU OOM
lr0: 0.01
# ... see config.yaml for all options
```

### Model Selection

- **yolov8n-seg.pt**: Nano (fastest, lowest accuracy)
- **yolov8s-seg.pt**: Small (balanced)
- **yolov8m-seg.pt**: Medium (recommended, good balance)
- **yolov8l-seg.pt**: Large (higher accuracy, slower)
- **yolov8x-seg.pt**: XLarge (best accuracy, slowest)

## 📊 Training Outputs

### Model Files
- `best.pt`: Best model weights (highest mAP)
- `last.pt`: Last epoch checkpoint
- `model.onnx`: ONNX export for deployment
- `model_fp16.onnx`: FP16 quantized ONNX (faster inference)

### Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU=0.5 (target: >0.85)
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision/Recall**: Per-class and overall metrics
- **Confusion Matrix**: Classification performance
- **PR Curves**: Precision-Recall curves

### Custom Metrics
- **Root Count**: Average roots detected per image
- **Area Coverage**: Total root area coverage percentage
- **Size Distribution**: Small/medium/large root classification

### Visualizations
- Training loss curves
- Validation metrics progression
- Confusion matrices
- Sample predictions with masks
- Before/after comparison images

### Data Files
- `predictions.csv`: All predictions with bboxes, masks, and confidence scores
- `low_confidence_predictions.csv`: Images flagged for re-annotation (active learning)

## 🔧 Advanced Features

### Root-Specific Preprocessing

The pipeline includes augmentations optimized for root detection:
- **HSV augmentation**: Handles lighting variations in field conditions
- **Mosaic augmentation**: Helps with overlapping roots
- **Scale augmentation**: Handles varying root sizes
- **No rotation**: Preserves root orientation (gravity matters)

### Active Learning

Low-confidence predictions are automatically flagged:
- Images with average confidence < 0.5
- Images with minimum confidence < 0.35
- Exported to CSV for review and re-annotation

### Custom Metrics

Root-specific metrics calculated:
```python
- Average roots per image
- Total area coverage (pixels)
- Average root area
- Size distribution (small/medium/large)
```

### Model Comparison

Baseline setup for comparing YOLOv8 with Mask R-CNN:
- Same dataset splits
- Same evaluation metrics
- Speed/accuracy tradeoff analysis

## 📈 Performance Targets

- **mAP@0.5**: > 0.85 (target)
- **Inference Speed**: 30+ FPS (real-time requirement)
- **Model Size**: < 100MB (for edge deployment)

## 🐛 Troubleshooting

### GPU Out of Memory (OOM)
- Reduce `batch` size in `config.yaml` (try 8, 4, or 2)
- Reduce `imgsz` (try 512 or 416)
- Enable gradient checkpointing

### Missing Files
- Verify dataset zip file name matches in notebook
- Check Google Drive mount path
- Ensure `data.yaml` exists in dataset root

### Low mAP
- Increase training epochs
- Adjust learning rate (`lr0`)
- Add more training data
- Try larger model (yolov8l-seg.pt or yolov8x-seg.pt)
- Review low-confidence predictions for annotation quality

### Corrupt Annotations
- Run validation cell to check image-annotation pairs
- Fix missing labels or images
- Verify YOLOv8 annotation format (normalized coordinates)

## 📝 Dataset Statistics

After running the validation cell, you'll see:
- Train/validation/test split counts
- Image-annotation pair validation
- Missing files report

## 🔄 Retraining with New Data

1. Update dataset zip file in Google Drive
2. Update `DATASET_ZIP` variable in notebook
3. Re-run data preparation cells
4. Training will start from pretrained weights (transfer learning)

## 📦 Model Deployment

### ONNX Runtime

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load model
session = ort.InferenceSession('model.onnx', 
                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Preprocess image
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

# Run inference
outputs = session.run(None, {'images': img})
```

### PyTorch

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('test_image.jpg', conf=0.25)
```

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{sweetpotato_yolov8,
  title = {Sweet Potato Root Detection with YOLOv8},
  author = {Tuskegee AIFARMS},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## 🤝 Contributing

This pipeline is part of the Tuskegee AIFARMS agricultural AI research initiative. For questions or contributions, please contact the research team.

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Ultralytics YOLOv8 team
- Roboflow for dataset management
- Tuskegee AIFARMS research team

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Production Ready
