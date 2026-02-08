# Quick Start Guide - Local Testing

This guide will help you test the training pipeline locally before using it in Google Colab.

## Prerequisites

1. **Python 3.8+** installed
2. **CUDA-capable GPU** (recommended) or CPU (will be slow)
3. **Dataset**: Your `Sweetpotato_roots.v2i.yolov8 (1).zip` file

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision ultralytics opencv-python matplotlib seaborn pandas numpy tqdm pyyaml onnx onnxruntime
```

## Step 2: Prepare Your Dataset

You have two options:

### Option A: Use Zip File
Place your dataset zip file in the project directory:
```
farm-photo-outliner/
├── Sweetpotato_roots.v2i.yolov8 (1).zip  ← Place it here
├── sweetpotato_yolov8_training.ipynb
├── config.yaml
└── ...
```

### Option B: Use Extracted Dataset
If you've already extracted the dataset, note the path and update it in the notebook.

## Step 3: Open the Notebook

Open `sweetpotato_yolov8_training.ipynb` in:
- **Jupyter Notebook**: `jupyter notebook`
- **Jupyter Lab**: `jupyter lab`
- **VS Code**: Open the .ipynb file directly
- **PyCharm**: Open the .ipynb file

## Step 4: Update Configuration

In the notebook, find the **"Configuration paths"** cell (Cell 6) and update:

```python
# For local mode, update these:
DATASET_ZIP = 'Sweetpotato_roots.v2i.yolov8 (1).zip'  # Or path to your zip
# OR
DATASET_DIR_EXTRACTED = r'C:\path\to\extracted\dataset'  # If already extracted
```

## Step 5: Run the Notebook

Run cells sequentially:
1. **Cell 2**: Setup (detects local vs Colab mode)
2. **Cell 3**: Check dependencies
3. **Cell 4**: Import libraries
4. **Cell 6**: Configure paths (update here!)
5. **Cell 7**: Extract/load dataset
6. **Cell 8**: Verify dataset structure
7. **Cell 9**: Validate annotations
8. Continue with training...

## What to Expect

### Local Mode Detection
The notebook automatically detects if it's running locally or in Colab:
- ✅ **Local mode**: No Google Drive needed, uses local paths
- ✅ **Colab mode**: Mounts Google Drive, uses `/content/` paths

### Output Locations
All outputs will be saved to:
```
./sweetpotato_project/
├── runs/
│   └── segment/
│       └── sweetpotato_exp/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── predictions.csv
│           └── ...
└── outputs/
    └── predictions/
```

## Troubleshooting

### "Missing packages" error
```bash
pip install -r requirements.txt
```

### "Dataset not found" error
- Check that the zip file is in the correct location
- Update `DATASET_ZIP` path in the configuration cell
- Or set `DATASET_DIR_EXTRACTED` to point to your extracted dataset

### GPU not detected
- Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Training will work on CPU but will be very slow

### Out of Memory (OOM)
- Reduce `batch` size in `config.yaml` (try 8, 4, or 2)
- Reduce `imgsz` (try 512 or 416)

## Testing Without Full Training

To quickly test the pipeline without full training:

1. In `config.yaml`, set:
   ```yaml
   epochs: 5  # Just 5 epochs for testing
   batch: 4  # Smaller batch
   ```

2. Run only the data preparation and validation cells first

3. If everything works, increase epochs for full training

## Next Steps

Once local testing is successful:
1. The same notebook will work in Google Colab
2. Just upload the notebook to Colab
3. Upload your dataset to Google Drive
4. Run all cells - it will automatically detect Colab mode

## Notes

- The notebook is **dual-mode**: works both locally and in Colab
- All paths are automatically adjusted based on environment
- No code changes needed when switching between local and Colab
