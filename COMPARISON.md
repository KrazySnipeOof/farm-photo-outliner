# Model Comparison: YOLOv8 vs Mask R-CNN

This document outlines how to set up a comparison between YOLOv8 and Mask R-CNN for sweet potato root detection.

## Overview

Both models will be trained on the same dataset with identical train/val/test splits to ensure fair comparison.

## Metrics for Comparison

### Accuracy Metrics
- **mAP@0.5** (bbox): Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95** (bbox): Mean Average Precision across IoU thresholds
- **mAP@0.5** (mask): Mask segmentation mAP at IoU=0.5
- **mAP@0.5:0.95** (mask): Mask segmentation mAP across IoU thresholds
- **Precision/Recall**: Per-class and overall metrics

### Speed Metrics
- **Inference FPS**: Frames per second on test set
- **Training Time**: Total training time
- **Model Size**: File size on disk

### Resource Usage
- **GPU Memory**: Peak memory usage during training
- **CPU Usage**: CPU utilization during inference

## YOLOv8 Setup

Already implemented in `sweetpotato_yolov8_training.ipynb`.

**Key Features:**
- Fast training and inference
- Good balance of speed and accuracy
- Easy deployment (ONNX export)
- Real-time capable (30+ FPS)

## Mask R-CNN Setup

### Installation

```bash
# Install Detectron2 (PyTorch-based Mask R-CNN implementation)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
```

### Training Script Template

```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register dataset (convert YOLOv8 format to COCO)
register_coco_instances("sweetpotato_train", {}, "train_annotations.json", "train/images")
register_coco_instances("sweetpotato_val", {}, "val_annotations.json", "valid/images")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sweetpotato_train",)
cfg.DATASETS.TEST = ("sweetpotato_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # sweetpotato_root, background

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### Format Conversion

YOLOv8 format needs to be converted to COCO format for Mask R-CNN:

```python
# Convert YOLOv8 annotations to COCO format
# (Implementation needed - can use roboflow or custom script)
```

## Comparison Results Template

| Metric | YOLOv8 | Mask R-CNN | Winner |
|--------|--------|------------|--------|
| mAP@0.5 (bbox) | - | - | - |
| mAP@0.5:0.95 (bbox) | - | - | - |
| mAP@0.5 (mask) | - | - | - |
| mAP@0.5:0.95 (mask) | - | - | - |
| Inference FPS | - | - | - |
| Training Time | - | - | - |
| Model Size | - | - | - |
| GPU Memory (peak) | - | - | - |

## Expected Trade-offs

### YOLOv8 Advantages
- **Speed**: Much faster inference (30+ FPS vs ~5-10 FPS)
- **Simplicity**: Easier to train and deploy
- **Size**: Smaller model files
- **Real-time**: Suitable for real-time applications

### Mask R-CNN Advantages
- **Accuracy**: Potentially higher mAP (especially mask mAP)
- **Mature**: Well-established architecture
- **Research**: More research papers and benchmarks

## Recommendations

1. **For Real-time Applications**: Use YOLOv8
   - Field deployment, harvesting automation
   - Requires 30+ FPS

2. **For Maximum Accuracy**: Compare both, choose based on results
   - If Mask R-CNN is significantly better (>5% mAP), use it
   - If similar, prefer YOLOv8 for speed

3. **For Edge Deployment**: Use YOLOv8
   - Better ONNX support
   - Smaller model size
   - Lower computational requirements

## Implementation Notes

1. **Same Data Splits**: Ensure identical train/val/test splits
2. **Same Evaluation**: Use same IoU thresholds and metrics
3. **Same Hardware**: Run on same GPU for fair speed comparison
4. **Multiple Runs**: Average results over 3-5 runs for statistical significance

## Next Steps

1. Convert YOLOv8 dataset to COCO format
2. Train Mask R-CNN with same hyperparameters (adjusted for architecture)
3. Evaluate both models on same test set
4. Compare metrics and choose best model for use case
