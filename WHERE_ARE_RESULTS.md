# Where to See Your Training Results

## 1. Main results folder (weights + training plots)

Open this folder in **File Explorer**:

```
C:\Users\kensm\farm-photo-outliner\sweetpotato_project\runs\segment\sweetpotato_exp
```

**If that folder is empty or doesn’t exist**, your run may have been saved under the project root. Try:

```
C:\Users\kensm\farm-photo-outliner\runs\segment\sweetpotato_project\runs\segment\sweetpotato_exp
```

**How to know for sure:** In the notebook, look at the **training cell** output for the line that says **"Results saved to …"** or **"Best model saved to …"**. That path is your main results folder.

**Inside that folder you’ll find:**

| File / folder      | What it is |
|--------------------|------------|
| **weights/best.pt** | Best model checkpoint — use this for inference. |
| **weights/last.pt** | Last epoch checkpoint. |
| **results.png**     | Training loss and metrics over epochs. |
| **confusion_matrix.png** | Confusion matrix. |
| **labels.jpg**     | Sample of training labels. |
| **args.yaml**      | Training arguments used. |

---

## 2. Validation curves and images (already in your repo)

These folders exist and contain validation metrics and sample images:

```
C:\Users\kensm\farm-photo-outliner\runs\segment\val
C:\Users\kensm\farm-photo-outliner\runs\segment\val2
```

**What’s inside:**

- **confusion_matrix.png** — Confusion matrix.
- **confusion_matrix_normalized.png** — Normalized version.
- **BoxPR_curve.png**, **MaskPR_curve.png** — Precision–recall curves (box and mask).
- **BoxF1_curve.png**, **BoxP_curve.png**, **BoxR_curve.png** — Box metrics.
- **MaskF1_curve.png**, **MaskP_curve.png**, **MaskR_curve.png** — Mask metrics.
- **val_batch0_labels.jpg** — Ground-truth labels for a validation batch.
- **val_batch0_pred.jpg** — Model predictions for the same batch.

Open these images to inspect validation performance and curves.

---

## 3. Inference outputs (after you run the inference cells)

After you run the inference and visualization cells in the notebook, predictions are saved here:

```
C:\Users\kensm\farm-photo-outliner\sweetpotato_project\outputs\predictions
```

You’ll find:

- Annotated images (masks/boxes overlaid).
- **predictions.csv** — One row per detection (image name, bbox, confidence, class, mask polygon).

---

## 4. Packed zip (after the “Download results” cell)

When you run the cell that packages and downloads results, it creates a zip file. On a **local** run the path is typically:

```
C:\Users\kensm\farm-photo-outliner\sweetpotato_project\sweetpotato_training_results.zip
```

The cell also prints the exact path when it finishes.

---

## Quick summary

| What you want           | Where to look |
|-------------------------|----------------|
| Best model (best.pt)    | `sweetpotato_project\runs\segment\sweetpotato_exp\weights\` or path from training output. |
| Loss and metric curves  | Same folder: **results.png**; or **runs\segment\val** and **val2** for validation curves. |
| Confusion matrix        | **runs\segment\val\confusion_matrix.png** (and val2). |
| Predicted vs ground truth | **runs\segment\val\val_batch0_pred.jpg** and **val_batch0_labels.jpg**. |
| Test predictions + CSV  | **sweetpotato_project\outputs\predictions** (after inference cells). |
| Everything in one zip   | **sweetpotato_project\sweetpotato_training_results.zip** (after download cell). |

**Easiest:** Open **`C:\Users\kensm\farm-photo-outliner\runs\segment`** in File Explorer and go into **val** or **val2** to see the curves and confusion matrix right away.
