"""
Inference script for language-conditioned crack segmentation model.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import LanguageConditionedSegmentationModel
from dataset import CrackSegmentationDataset


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        clip_model_name = config.get('clip_model', 'ViT-B/32')
    else:
        # Fallback to default if config not found
        print("Warning: No config found in checkpoint, using default CLIP model 'ViT-B/32'")
        clip_model_name = 'ViT-B/32'
    
    # Create model
    device_str = "cuda" if device.type == "cuda" else "cpu"
    model = LanguageConditionedSegmentationModel(
        clip_model_name=clip_model_name,
        device=device_str
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (CLIP model: {clip_model_name})")
    if 'val_dice' in checkpoint:
        print(f"Model validation Dice score: {checkpoint['val_dice']:.4f}")
    
    return model


def preprocess_image(image_path: str, target_size: Optional[tuple] = None) -> torch.Tensor:
    """
    Preprocess image for inference.
    
    Args:
        image_path: Path to input image
        target_size: Optional (height, width) to resize to
    
    Returns:
        Preprocessed image tensor [1, 3, H, W] normalized to [0, 1]
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # Resize if target size specified
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Convert to numpy and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Convert to tensor: [H, W, 3] -> [3, H, W]
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    # Add batch dimension: [3, H, W] -> [1, 3, H, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, original_size


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    prompt: str,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        prompt: Text prompt for segmentation
        device: Device to run inference on
        threshold: Threshold for binary mask (default: 0.5)
    
    Returns:
        Binary mask as numpy array [H, W] with values {0, 1}
    """
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Use channels_last format if CUDA
    if device.type == "cuda":
        image_tensor = image_tensor.to(memory_format=torch.channels_last)
    
    # Run inference
    with torch.no_grad():
        logits = model(image_tensor, prompt)
        
        # Apply sigmoid and threshold
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()
    
    # Convert to numpy: [1, 1, H, W] -> [H, W]
    pred_mask_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8)
    
    return pred_mask_np


def save_prediction(
    mask: np.ndarray,
    output_path: str,
    original_size: Optional[tuple] = None
):
    """
    Save prediction mask as PNG image.
    
    Args:
        mask: Binary mask [H, W] with values {0, 1}
        output_path: Path to save output image
        original_size: Optional (width, height) to resize mask to
    """
    # Resize mask to original size if specified
    if original_size is not None:
        mask_pil = Image.fromarray(mask * 255, mode='L')
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
        mask = np.array(mask_pil) / 255
    
    # Convert to {0, 255} for PNG
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Save as PNG
    mask_image = Image.fromarray(mask_uint8, mode='L')
    mask_image.save(output_path)
    print(f"Saved prediction mask to {output_path}")


def compute_metrics_numpy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute Dice and IoU metrics from numpy arrays.
    
    Args:
        pred_mask: Predicted binary mask [H, W] with values {0, 1}
        gt_mask: Ground truth binary mask [H, W] with values {0, 1}
    
    Returns:
        Dictionary with 'dice' and 'iou' scores
    """
    # Ensure masks are binary
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Compute intersection and union
    intersection = np.sum(pred_flat * gt_flat)
    pred_sum = np.sum(pred_flat)
    gt_sum = np.sum(gt_flat)
    union = np.sum(np.clip(pred_flat + gt_flat, 0, 1))
    
    # Compute Dice
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-6)
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return {
        'dice': float(dice),
        'iou': float(iou)
    }


def load_ground_truth_mask(image_path: Path, annotations_path: Path) -> Optional[np.ndarray]:
    """
    Load ground truth mask for an image from COCO annotations.
    
    Args:
        image_path: Path to the image file
        annotations_path: Path to COCO annotations JSON file
    
    Returns:
        Ground truth mask as numpy array [H, W] or None if not found
    """
    if not annotations_path.exists():
        return None
    
    try:
        # Load COCO annotations
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Find image in annotations
        image_name = image_path.name
        image_info = None
        for img in coco_data['images']:
            if img['file_name'] == image_name:
                image_info = img
                break
        
        if image_info is None:
            return None
        
        # Get annotations for this image
        annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] == image_info['id']
        ]
        
        if len(annotations) == 0:
            return None
        
        # Create dataset instance to use its mask conversion methods
        # We'll create a temporary dataset just to use the mask conversion
        from dataset import CrackSegmentationDataset
        temp_dataset = CrackSegmentationDataset(
            annotations_path=str(annotations_path),
            images_dir=str(image_path.parent),
            split="valid"
        )
        
        # Convert annotations to mask
        height = image_info['height']
        width = image_info['width']
        mask = temp_dataset._annotations_to_mask(annotations, height, width)
        
        return mask
        
    except Exception as e:
        print(f"  Warning: Could not load ground truth mask: {e}")
        return None


def visualize_prediction(
    image_path: str,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.5,
    gt_mask: Optional[np.ndarray] = None
):
    """
    Create visualization overlay of prediction on original image.
    If gt_mask is provided, creates side-by-side comparison.
    
    Args:
        image_path: Path to original image
        mask: Binary mask [H, W] with values {0, 1}
        output_path: Path to save visualization
        alpha: Transparency of overlay (default: 0.5)
        gt_mask: Optional ground truth mask for side-by-side comparison
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    image_height, image_width = image_array.shape[:2]
    
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    
    # Resize mask to match image size if needed
    mask_height, mask_width = mask.shape[:2]
    if (mask_height, mask_width) != (image_height, image_width):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((image_width, image_height), Image.NEAREST)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
    
    # Ensure mask is in [0, 1] range
    mask = np.clip(mask, 0, 1)
    
    def create_overlay(img_array, msk, title=""):
        """Helper function to create overlay visualization."""
        overlay = img_array.astype(np.float32).copy()
        mask_3d = np.stack([msk] * 3, axis=-1)  # [H, W, 3]
        red_overlay = np.array([255, 0, 0], dtype=np.float32)  # Red color
        overlay = np.where(mask_3d > 0.5, 
                          overlay * (1 - alpha) + red_overlay * alpha,
                          overlay)
        return overlay.astype(np.uint8)
    
    # Create prediction overlay
    pred_overlay = create_overlay(image_array, mask)
    
    # If ground truth mask provided, create side-by-side comparison
    if gt_mask is not None:
        # Process ground truth mask
        if gt_mask.ndim > 2:
            gt_mask = gt_mask.squeeze()
        gt_mask_height, gt_mask_width = gt_mask.shape[:2]
        if (gt_mask_height, gt_mask_width) != (image_height, image_width):
            gt_mask_pil = Image.fromarray((gt_mask * 255).astype(np.uint8), mode='L')
            gt_mask_pil = gt_mask_pil.resize((image_width, image_height), Image.NEAREST)
            gt_mask = np.array(gt_mask_pil, dtype=np.float32) / 255.0
        gt_mask = np.clip(gt_mask, 0, 1)
        
        # Create ground truth overlay
        gt_overlay = create_overlay(image_array, gt_mask)
        
        # Create side-by-side image
        combined = np.hstack([pred_overlay, gt_overlay])
        
        # Add text labels (simple approach - could use PIL ImageDraw for better text)
        # For now, save as is - labels can be added with image editing tools if needed
        combined_image = Image.fromarray(combined)
        combined_image.save(output_path)
        print(f"Saved side-by-side visualization (Prediction | Ground Truth) to {output_path}")
    else:
        # Single image with prediction overlay
        overlay_image = Image.fromarray(pred_overlay)
        overlay_image.save(output_path)
        print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on crack segmentation model")
    
    # Input arguments (either --image or --folder must be provided)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, default=None,
                            help="Path to input image (single image inference)")
    input_group.add_argument("--folder", type=str, default=None,
                            help="Path to folder containing images (batch inference)")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth file)")
    
    # Batch inference arguments
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to process from folder (default: all images)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for batch inference results (default: same as input folder)")
    
    # Optional arguments
    parser.add_argument("--prompt", type=str, default="segment crack",
                        help="Text prompt for segmentation (default: 'segment crack')")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output mask (single image mode, default: <image_name>_mask.png)")
    parser.add_argument("--visualization", type=str, default=None,
                        help="Path to save visualization overlay (single image mode, optional)")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualization overlays in batch mode")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary mask (default: 0.5)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device to use (default: auto)")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Single image mode
    if args.image:
        print(f"\n=== Single Image Inference ===")
        print(f"Loading image: {args.image}")
        image_tensor, original_size = preprocess_image(args.image)
        print(f"Image size: {original_size[0]}x{original_size[1]}")
        
        # Run inference
        print(f"\nRunning inference with prompt: '{args.prompt}'...")
        mask = predict(model, image_tensor, args.prompt, device, args.threshold)
        
        # Calculate mask statistics
        mask_area = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage = (mask_area / total_pixels) * 100
        print(f"Predicted mask coverage: {coverage:.2f}% ({mask_area}/{total_pixels} pixels)")
        
        # Determine output path
        if args.output is None:
            image_path = Path(args.image)
            output_path = image_path.parent / f"{image_path.stem}_mask.png"
        else:
            output_path = Path(args.output)
        
        # Save prediction mask
        save_prediction(mask, str(output_path), original_size)
        
        # Save visualization if requested
        if args.visualization:
            visualize_prediction(args.image, mask, args.visualization)
        elif args.visualization is None:
            # Auto-generate visualization path
            image_path = Path(args.image)
            vis_path = image_path.parent / f"{image_path.stem}_overlay.png"
            visualize_prediction(args.image, mask, str(vis_path))
        
        print("\nInference completed!")
    
    # Batch inference mode
    elif args.folder:
        print(f"\n=== Batch Inference Mode ===")
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"Error: Folder not found: {args.folder}")
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ]
        
        if len(image_files) == 0:
            print(f"Error: No image files found in {args.folder}")
            return
        
        # Sort files for consistent ordering
        image_files.sort()
        
        # Limit number of images if specified
        if args.num_images is not None:
            image_files = image_files[:args.num_images]
        
        print(f"Found {len(image_files)} image(s) to process")
        
        # Create output directory
        if args.output_dir is None:
            output_dir = folder_path / "predictions"
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_visualizations:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to find COCO annotations file for ground truth masks
        annotations_path = None
        # Check common locations for annotations
        possible_annotation_paths = [
            folder_path / "_annotations.coco.json",
            folder_path.parent / folder_path.name / "_annotations.coco.json",
            folder_path.parent.parent / folder_path.name / "_annotations.coco.json"
        ]
        for ann_path in possible_annotation_paths:
            if ann_path.exists():
                annotations_path = ann_path
                print(f"Found annotations file: {annotations_path}")
                break
        
        if annotations_path is None:
            print("Warning: No COCO annotations file found. Metrics and ground truth comparison will be skipped.")
        
        # Process each image
        print(f"\nProcessing images with prompt: '{args.prompt}'...")
        print(f"Output directory: {output_dir}")
        
        total_coverage = 0.0
        total_dice = 0.0
        total_iou = 0.0
        metrics_count = 0
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")
            
            try:
                # Preprocess image
                image_tensor, original_size = preprocess_image(str(image_file))
                
                # Run inference
                mask = predict(model, image_tensor, args.prompt, device, args.threshold)
                
                # Calculate mask statistics
                mask_area = np.sum(mask)
                total_pixels = mask.shape[0] * mask.shape[1]
                coverage = (mask_area / total_pixels) * 100
                total_coverage += coverage
                
                # Load ground truth mask if available
                gt_mask = None
                if annotations_path:
                    gt_mask = load_ground_truth_mask(image_file, annotations_path)
                
                # Compute metrics if ground truth available
                metrics_str = ""
                if gt_mask is not None:
                    metrics = compute_metrics_numpy(mask, gt_mask)
                    total_dice += metrics['dice']
                    total_iou += metrics['iou']
                    metrics_count += 1
                    metrics_str = f" | Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}"
                
                print(f"  Coverage: {coverage:.2f}% ({mask_area}/{total_pixels} pixels){metrics_str}")
                
                # Save prediction mask
                output_path = output_dir / f"{image_file.stem}_mask.png"
                save_prediction(mask, str(output_path), original_size)
                
                # Save visualization if requested
                if args.save_visualizations:
                    vis_path = vis_dir / f"{image_file.stem}_overlay.png"
                    visualize_prediction(str(image_file), mask, str(vis_path), gt_mask=gt_mask)
                
            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
                continue
        
        # Print summary
        avg_coverage = total_coverage / len(image_files) if len(image_files) > 0 else 0.0
        print(f"\n=== Batch Inference Summary ===")
        print(f"Processed: {len(image_files)} image(s)")
        print(f"Average coverage: {avg_coverage:.2f}%")
        
        if metrics_count > 0:
            avg_dice = total_dice / metrics_count
            avg_iou = total_iou / metrics_count
            print(f"Average Dice: {avg_dice:.4f}")
            print(f"Average IoU (mIoU): {avg_iou:.4f}")
        else:
            print("Metrics: Not available (no ground truth annotations found)")
        
        print(f"Output directory: {output_dir}")
        print("Inference completed!")


if __name__ == "__main__":
    main()

