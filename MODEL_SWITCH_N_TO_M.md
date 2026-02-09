# YOLOv8: Switch from n (nano) to m (medium) segmentation

This project uses the **YOLOv8 m-seg** (medium) checkpoint for sweet potato root segmentation instead of the smaller **n (nano)** variant. The m model has more parameters and channels, giving higher capacity for finer, more accurate masks. Training and inference code is unchanged; only the checkpoint name and comments were updated.

---

## 1. Model-loading code: BEFORE and AFTER

### Notebook (training)

**BEFORE (if you had been using n):**
```python
# Initialize YOLOv8 segmentation model
model_name = default_config['model']   # e.g. 'yolov8n-seg.pt'
model = YOLO(model_name)
```

**AFTER (current):**
```python
# Initialize YOLOv8 segmentation model (m-seg = medium; more parameters/channels than n for finer masks)
model_name = default_config['model']   # 'yolov8m-seg.pt'
model = YOLO(model_name)
```

- `model_name` comes from `default_config['model']`, which is set from **config.yaml** or the notebook default.
- `model.train(...)` and `model.predict(...)` are unchanged; same arguments (conf, iou, device, etc.).

### Notebook (evaluation fallback)

**BEFORE:**
```python
model = YOLO(model_name)  # Fallback to pretrained
```

**AFTER:**  
Same line; `model_name` is already `yolov8m-seg.pt` from config. No code change; only the config/default ensures m-seg.

### inference.py

**BEFORE:**
```python
model = YOLO(model_path)
```

**AFTER:**  
Same line. The script loads whatever `.pt` you pass with `--model` (e.g. a trained `best.pt` from m-seg). A comment was added at the top of the file stating that the pipeline is designed for YOLOv8 m-seg trained models.

---

## 2. Config / YAML changes

### config.yaml

**BEFORE (if you had n):**
```yaml
# Model selection
# Options: yolov8n-seg.pt (nano), ...
model: yolov8n-seg.pt
```

**AFTER (current):**
```yaml
# YOLOv8 Training Configuration for Sweet Potato Root Detection
# Tuskegee AIFARMS Agricultural AI Research
# We use yolov8m-seg (medium) for higher capacity and finer masks than n (nano).

# Model selection
# Options: yolov8n-seg.pt (nano), yolov8s-seg.pt (small), yolov8m-seg.pt (medium),
#          yolov8l-seg.pt (large), yolov8x-seg.pt (xlarge)
model: yolov8m-seg.pt
```

- No change to dataset paths, `data.yaml`, or logging locations.
- Input channels, number of classes, and segmentation behavior are defined by the dataset and task; the m-seg checkpoint matches the same interface as n-seg.

### Notebook default_config (in-code fallback)

**BEFORE (if you had n):**
```python
default_config = {
    'model': 'yolov8n-seg.pt',  # ...
}
```

**AFTER (current):**
```python
# Default configuration (yolov8m-seg = medium; more capacity than n for finer masks)
default_config = {
    'model': 'yolov8m-seg.pt',  # Options: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, ...
}
```

- If `config.yaml` is present, it overrides this default, so keeping `model: yolov8m-seg.pt` in **config.yaml** is enough to use m-seg everywhere.

---

## 3. Resource impact (VRAM / compute)

- **yolov8n-seg**: ~3.4M params; lowest VRAM and fastest.
- **yolov8m-seg**: ~25.9M params; more VRAM and compute than n.

**Practical impact:**

- **Training:** For the same batch size (e.g. 16) and image size (640), expect roughly **1.5–2× more VRAM** with m than with n. On an 8 GB GPU (e.g. RTX 4070 Laptop), batch=16 may still run; if you see OOM, reduce to **batch=8** or **batch=4** (no other code changes needed).
- **Inference:** m-seg is slower than n-seg per image but still real-time on a decent GPU; expect a lower FPS (e.g. still well above 30 FPS on modern GPUs for 640 input).
- **Batch size:** Not changed in this switch. Reduce only if you hit out-of-memory errors.

---

## 4. Summary

| Location            | Change |
|--------------------|--------|
| **config.yaml**    | `model: yolov8m-seg.pt` + top comment about m-seg for capacity. |
| **Notebook**       | Default `default_config['model']` = `'yolov8m-seg.pt'`; comments in config cell and model-init cell; first markdown explains m-seg choice. |
| **inference.py**   | Top-of-file comment that pipeline uses m-seg; `YOLO(model_path)` unchanged. |
| **Train/predict**  | No changes to `model.train(...)` or `model.predict(...)` arguments. |
| **Dataset/paths**  | No changes to dataset paths, `data.yaml`, or logging directories. |

All model loads now use the **m** segmentation checkpoint for more capacity and finer masks; task remains segmentation throughout.
