# Project Summary & Next Steps  
# Sweet Potato Root Detection with YOLOv8

This document explains **everything that has happened** in this project and **exactly what to do next**.

---

## Part 1: What Was Built (The Pipeline)

### 1.1 Goal

A **production-ready training pipeline** for sweet potato root **detection and instance segmentation** using Ultralytics YOLOv8, for:

- Tuskegee AIFARMS agricultural AI research  
- Yield estimation and quality assessment  
- Real-time root detection (e.g. harvesting automation)  
- Dataset: Roboflow **Sweetpotato_roots** (YOLOv8 format, v2)

### 1.2 What Exists in the Repo

| Item | Purpose |
|------|--------|
| **sweetpotato_yolov8_training.ipynb** | Main pipeline: setup → data → train → evaluate → infer → export. Works in **Google Colab** and **locally**. |
| **config.yaml** | All training hyperparameters (model, epochs, batch, augmentation, etc.). Single place to tune. |
| **requirements.txt** | Exact package versions for reproducibility. |
| **README.md** | Full setup, dataset structure, usage, troubleshooting. |
| **QUICK_START_LOCAL.md** | Short guide to run the notebook on your machine. |
| **inference.py** | Standalone script to run a trained model on new images (no notebook). |
| **COMPARISON.md** | How to compare YOLOv8 vs Mask R-CNN on the same data. |
| **.gitignore** | Ignores models, datasets, and large outputs. |

### 1.3 What the Notebook Does (Step by Step)

1. **Setup**  
   - Detects Colab vs local.  
   - Colab: mounts Drive, installs packages.  
   - Local: checks/installs packages, no Drive.

2. **Data**  
   - Finds dataset zip or uses an already-extracted folder.  
   - Unzips if needed, flattens nested folders.  
   - Verifies `train/`, `valid/`, `test/` and `data.yaml`.  
   - Validates that every image has a matching label file.

3. **Training**  
   - Loads `config.yaml` (or defaults).  
   - Loads a YOLOv8 **segmentation** model (e.g. `yolov8m-seg.pt`).  
   - Trains on your data with augmentations, checkpoints every 10 epochs, OOM handling (smaller batch).

4. **Evaluation**  
   - Loads best checkpoint, runs validation and test.  
   - Prints mAP (box and mask), precision, recall.  
   - Can show confusion matrix, custom metrics (root count, area, size distribution), and active-learning flags (low-confidence images).

5. **Inference & export**  
   - Runs predictions on the test set.  
   - Saves annotated images and a **CSV** (bbox + mask polygons).  
   - Exports **ONNX** (and optionally TorchScript/FP16) for deployment.

6. **Download / save**  
   - Colab: zips results and triggers download.  
   - Local: writes zip and prints where it is.

---

## Part 2: What Has Happened (Chronology)

### 2.1 Pipeline Created

- The notebook, config, requirements, README, inference script, and comparison doc were created so you could:
  - Run in Colab (Drive + GPU) or locally.
  - Train once and get best model, metrics, plots, CSV, and ONNX.

### 2.2 Local Run: “Run Here Before Colab”

- You wanted to run the notebook **locally** first.
- The notebook was updated to be **dual-mode**:
  - **Colab**: mount Drive, install deps, use `/content/` paths.
  - **Local**: no Drive; use paths like `./sweetpotato_project` and your dataset path; optionally auto-install missing packages.

### 2.3 Import Error: `ModuleNotFoundError: No module named 'torch'`

- On first run, the **import cell** failed because PyTorch and other deps were not installed in the environment the notebook was using.
- **Changes made:**
  - **Cell 3 (dependencies):** In local mode, it now **checks** for required packages and, if any are missing, **installs** them (from `requirements.txt` or one-by-one). So running Cell 3 first can fix missing packages.
  - **Cell 4 (imports):** Wrapped in try/except so you get a clear message: “Run the previous cell to install dependencies” or “Install with: pip install -r requirements.txt”.
- You (or the environment) also ran `pip install -r requirements.txt`; most packages installed. **ONNX** failed on Windows due to very long path names in the package; the notebook was updated so ONNX is **optional** (training and inference work without it; export to ONNX is skipped or handled gracefully if not installed).

### 2.4 Dataset and Config

- Your **actual dataset** (e.g. from `Sweetpotato_roots.v2i.yolov8 (1)`):
  - **Classes:** 3 — **Diseased**, **Healthy**, **Non-determined** (not 2).
  - **Format:** YOLOv8 segmentation (per-line: class_id + normalized polygon x,y points).
  - **Splits:** `train/`, `valid/`, `test/` with `images/` and `labels/` in each.
- **data.yaml** contains:
  - `nc: 3`
  - `names: ['Diseased', 'Healthy', 'Non-determined']`
  - Paths: `train/images`, `valid/images`, `test/images` (relative to dataset root).
- The log message **“Overriding model.yaml nc=80 with nc=3”** is **expected**: the base YOLOv8-seg model is COCO (80 classes); your dataset correctly overrides to 3 classes.

### 2.5 Training Run (Where You Are Now)

- You started training **locally** with:
  - **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU.  
  - **Software:** Ultralytics 8.4.12, Python 3.13.5, PyTorch 2.6.0.  
  - **Settings:** 100 epochs, batch 16, image size 640, from `config.yaml` (and notebook defaults).
- **Observed:**
  - Training **is running**: e.g. Epoch 33/100, GPU memory in use, loss values (box, seg, cls, dfl) updating.
  - **Validation metrics:** At epoch 33, **all** reported mAP (box and mask, P/R, mAP50, mAP50–95) were **0**.
  - **Warnings:** `crop_fraction` and `label_smoothing` deprecated in Ultralytics 8.4+.
- **Why mAP can be 0 at epoch 33:**
  - Segmentation often needs more epochs before mask quality is good enough to get non-zero mAP.
  - With a small validation set (e.g. 5 images, 7 instances), a few “wrong” epochs can keep mAP at 0; it may jump later.
  - If the **dataset path** used at training time is wrong (e.g. not the folder that contains `data.yaml` and `train/`/`valid/`/`test/`), validation could be wrong or empty; that can also show as zeros.

### 2.6 Config Cleanup

- **config.yaml** was updated to **remove** the deprecated options:
  - `crop_fraction`
  - `label_smoothing`  
- So future runs won’t rely on those; the deprecation warnings may still appear if the notebook passes them explicitly, but they don’t break training.

---

## Part 3: Current State (Summary)

- **Pipeline:** Complete and dual-mode (Colab + local).  
- **Environment:** Local run with RTX 4070; dependencies resolved (ONNX optional).  
- **Dataset:** 3-class sweet potato roots (Diseased / Healthy / Non-determined), YOLOv8 segmentation format.  
- **Training:** Started, reached at least epoch 33; losses updating; **validation mAP still 0**.  
- **Config:** Deprecated options removed from `config.yaml`.

---

## Part 4: Next Steps (In Order)

### 4.1 Let This Training Run Finish

- **Do not stop the current run** just because mAP is 0 at epoch 33.
- Let it run to **100 epochs** (or until early stopping if you enabled it).
- Segmentation mAP often starts to move in the **second half** of training (e.g. 50–70+).
- Check the **loss curves** in the run folder; if train/val loss are decreasing, the model is learning.

**Where results are saved (local):**

- Default: `./sweetpotato_project/runs/segment/sweetpotato_exp/`
  - `weights/best.pt` — best checkpoint by validation metric  
  - `weights/last.pt` — last epoch  
  - Plots (loss, metrics), confusion matrix, etc.

### 4.2 After Training Finishes: Run the Rest of the Notebook

- **Evaluation cells:** Load `best.pt`, run validation and test again, print mAP (box + mask), P/R.
- **Confusion matrix:** If the script saves it, display or open the image from the run folder.
- **Custom metrics:** Root count per image, area coverage, size distribution (if you have those cells).
- **Active learning:** Generate list/CSV of low-confidence images for re-annotation.
- **Inference:** Run prediction on the test set; save overlaid images and CSV (bbox + mask polygons).
- **Export:**  
  - Export **ONNX** (and TorchScript/FP16 if you use those).  
  - If ONNX isn’t installed (e.g. Windows path issue), you can skip export or install ONNX in a different way later.
- **Package results:** Zip `best.pt`, ONNX (if present), CSV, and key plots for your records.

### 4.3 If mAP Is Still 0 After 100 Epochs

Then we need to fix data and paths:

1. **Dataset path**  
   - When you start the notebook and run the “data” and “train” cells, the **current working directory** and the path passed to `model.train(data=...)` must be such that:
   - `data.yaml` is found.
   - Inside `data.yaml`, paths like `train/images` and `valid/images` are resolved **relative to the directory that contains `data.yaml`** (the dataset root).
   - Fix: Either set the notebook’s working directory to the dataset root, or put the **absolute path** of the dataset root in `data.yaml` under a `path:` key (if Ultralytics supports it for your version) and pass the correct `data.yaml` path to `model.train()`.

2. **Verify labels**  
   - Open a few `train/labels/*.txt` and `valid/labels/*.txt` files.  
   - Each line should be: `class_id x1 y1 x2 y2 x3 y3 ...` (normalized 0–1), with class_id in `0, 1, 2` for your 3 classes.  
   - If any class_id is ≥ 3 or the format is different, that will break training/validation.

3. **Re-run training**  
   - After fixing paths (and labels if needed), start a **new** training run (new run name or overwrite as you prefer) and run to 100 epochs again.

### 4.4 If mAP Becomes Non-Zero (Success Path)

- **Use best.pt** for inference on new images (notebook or `inference.py`).
- **Track metrics:** Note final mAP50 and mAP50-95 for box and mask; add them to README or a small results file.
- **Deploy:** Use ONNX (or TorchScript) for deployment; test inference speed (e.g. FPS) on your target hardware.
- **Iterate:** If you add more data (e.g. from active learning), re-train with the same pipeline and compare metrics.

### 4.5 Optional: Colab for Heavier Runs

- Upload the **same notebook** and dataset (e.g. zip in Drive).
- In the config cell, set paths for Colab (e.g. dataset zip path in Drive).
- Run all cells; Colab will use its GPU and Drive. Good for larger batches or if you want a clean environment.

### 4.6 Optional: Compare with Mask R-CNN

- Use **COMPARISON.md** and the same dataset (convert to COCO if needed) to train Mask R-CNN and compare mAP and speed vs YOLOv8.

---

## Part 5: Quick Reference

| Question | Answer |
|----------|--------|
| What is this project? | YOLOv8 segmentation pipeline for sweet potato roots (3 classes: Diseased, Healthy, Non-determined). |
| What’s in the notebook? | End-to-end: setup → data → train → evaluate → infer → export (Colab + local). |
| Why did imports fail at first? | Packages (e.g. torch) were not installed; Cell 3 now checks/installs them locally. |
| Why is ONNX optional? | It failed to install on Windows (path length); training doesn’t require it. |
| Why was mAP 0 at epoch 33? | Normal for segmentation to lag; small val set; possibly wrong dataset path. |
| What to do now? | Let training run to 100 epochs, then run evaluation/inference/export cells and check mAP. |
| Where are outputs? | Under `./sweetpotato_project/runs/segment/sweetpotato_exp/` (e.g. `weights/best.pt`). |
| If mAP stays 0? | Fix dataset path and labels, then re-train. |

---

## Part 6: Files You Care About Right Now

- **sweetpotato_yolov8_training.ipynb** — Run this; finish training, then run evaluation and export cells.  
- **config.yaml** — Change epochs, batch size, model size here if you want.  
- **PROJECT_SUMMARY_AND_NEXT_STEPS.md** (this file) — Full story and next steps.  
- **Dataset folder** — The one that contains `data.yaml`, `train/`, `valid/`, `test/`. Make sure the notebook points to it correctly.

If you tell me whether training has finished and what mAP you see after 100 epochs, I can give you the exact next steps (e.g. “run these cells” or “fix path X and re-train”).
