"""
Sweet Potato Root Detection - Inference Script
Standalone script for running inference on new images after training.

Usage:
    python inference.py --model best.pt --source path/to/images --output results/
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def run_inference(model_path, source, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run inference on images and save results.
    
    Args:
        model_path: Path to trained model (.pt file)
        source: Path to image(s) or directory
        output_dir: Directory to save results
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image paths
    if os.path.isfile(source):
        image_paths = [source]
    elif os.path.isdir(source):
        image_paths = [
            str(p) for p in Path(source).rglob('*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
    else:
        raise ValueError(f"Source path not found: {source}")
    
    print(f"Found {len(image_paths)} image(s)")
    
    # Run inference
    all_predictions = []
    
    for img_path in tqdm(image_paths, desc="Running inference"):
        # Predict
        results = model.predict(
            img_path,
            conf=conf_threshold,
            iou=iou_threshold,
            save=True,
            save_txt=True,
            save_conf=True,
            project=output_dir,
            name='predictions',
            exist_ok=True
        )
        
        # Extract predictions
        result = results[0]
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                bbox = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls = int(box.cls[0].cpu().item())
                
                # Get mask polygon if available
                mask_polygon = None
                if result.masks is not None and i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    # Convert mask to polygon
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        mask_polygon = largest_contour.reshape(-1, 2).tolist()
                
                all_predictions.append({
                    'image_name': os.path.basename(img_path),
                    'image_path': img_path,
                    'bbox_x1': float(bbox[0]),
                    'bbox_y1': float(bbox[1]),
                    'bbox_x2': float(bbox[2]),
                    'bbox_y2': float(bbox[3]),
                    'confidence': conf,
                    'class': cls,
                    'mask_polygon': str(mask_polygon) if mask_polygon else None
                })
    
    # Save predictions to CSV
    if all_predictions:
        df = pd.DataFrame(all_predictions)
        csv_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Predictions saved to: {csv_path}")
        print(f"✓ Total detections: {len(all_predictions)}")
        print(f"\nSummary:")
        print(f"  Images processed: {len(image_paths)}")
        print(f"  Average detections per image: {len(all_predictions) / len(image_paths):.2f}")
    else:
        print("\n⚠ No detections found. Try lowering confidence threshold.")
    
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Sweet Potato Root Detection - Inference'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image(s) or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='inference_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Run inference
    run_inference(
        args.model,
        args.source,
        args.output,
        args.conf,
        args.iou
    )


if __name__ == '__main__':
    main()
