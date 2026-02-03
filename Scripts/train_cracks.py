"""
Training script for language-conditioned crack segmentation.
"""

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from datetime import datetime

from dataset import CrackSegmentationDataset
from model import LanguageConditionedSegmentationModel
from losses import CombinedLoss, DiceLoss
from utils import load_prompts, sample_prompt


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Enable cuDNN benchmarking for better performance (after seed is set)
    torch.backends.cudnn.benchmark = True


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute Dice score and IoU.
    
    Args:
        predictions: Logits [B, 1, H, W]
        targets: Binary masks [B, 1, H, W] with values {0, 1}
    
    Returns:
        Dictionary with 'dice' and 'iou' scores
    """
    # Apply sigmoid and threshold
    pred_probs = torch.sigmoid(predictions)
    pred_binary = (pred_probs > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = targets.view(-1)
    
    # Compute Dice
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    
    # Compute IoU
    union = (pred_flat + target_flat).clamp(0, 1).sum()
    iou = intersection / (union + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item()
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    prompt_embeddings_cache: Dict[str, torch.Tensor],
    prompts: Dict[str, list],
    writer: SummaryWriter,
    global_step: int,
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    class_name: str = "crack"
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    # Individual loss components for logging
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    
    for batch_idx, (images, masks, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Convert to channels_last memory format for better performance on CUDA
        if use_amp:  # Only when using CUDA (use_amp is True only for CUDA)
            images = images.to(memory_format=torch.channels_last)
            masks = masks.to(memory_format=torch.channels_last)
        
        # Sample prompt for this batch and retrieve cached embedding
        prompt = sample_prompt(class_name, prompts, is_training=True)
        text_embedding = prompt_embeddings_cache[prompt].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model.forward_with_embedding(images, text_embedding)
                loss = criterion(logits, masks)
                # Compute individual loss components for logging
                bce_loss = bce_loss_fn(logits, masks)
                dice_loss = dice_loss_fn(logits, masks)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model.forward_with_embedding(images, text_embedding)
            loss = criterion(logits, masks)
            # Compute individual loss components for logging
            bce_loss = bce_loss_fn(logits, masks)
            dice_loss = dice_loss_fn(logits, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_dice_loss += dice_loss.item()
        num_batches += 1
        
        # Log every 10 iterations
        if (batch_idx + 1) % 10 == 0:
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            writer.add_scalar('train/bce_loss', bce_loss.item(), global_step)
            writer.add_scalar('train/dice_loss', dice_loss.item(), global_step)
            global_step += 1
            
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}, Prompt: {prompt}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_bce_loss = total_bce_loss / num_batches if num_batches > 0 else 0.0
    avg_dice_loss = total_dice_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'bce_loss': avg_bce_loss,
        'dice_loss': avg_dice_loss
    }, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prompt_embeddings_cache: Dict[str, torch.Tensor],
    use_amp: bool,
    class_name: str = "crack"
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    # Use fixed prompt for validation
    val_prompt = "segment crack"
    val_text_embedding = prompt_embeddings_cache[val_prompt].to(device)
    
    with torch.no_grad():
        for images, masks, labels in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Convert to channels_last memory format for better performance on CUDA
            if use_amp:  # Only when using CUDA (use_amp is True only for CUDA)
                images = images.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)
            
            # Forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model.forward_with_embedding(images, val_text_embedding)
                    loss = criterion(logits, masks)
            else:
                logits = model.forward_with_embedding(images, val_text_embedding)
                loss = criterion(logits, masks)
            
            # Compute metrics
            metrics = compute_metrics(logits, masks)
            
            total_loss += loss.item()
            total_dice += metrics['dice']
            total_iou += metrics['iou']
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'dice': avg_dice,
        'iou': avg_iou
    }


def save_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    prompt_embeddings_cache: Dict[str, torch.Tensor],
    use_amp: bool,
    num_samples: int = 5
):
    """Save prediction visualizations for a few validation samples."""
    model.eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    val_prompt = "segment crack"
    val_text_embedding = prompt_embeddings_cache[val_prompt].to(device)
    
    saved_count = 0
    
    with torch.no_grad():
        for images, masks, labels in dataloader:
            if saved_count >= num_samples:
                break
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Convert to channels_last memory format for better performance on CUDA
            if use_amp:  # Only when using CUDA (use_amp is True only for CUDA)
                images = images.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)
            
            # Forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model.forward_with_embedding(images, val_text_embedding)
            else:
                logits = model.forward_with_embedding(images, val_text_embedding)
            pred_probs = torch.sigmoid(logits)
            pred_binary = (pred_probs > 0.5).float()
            
            # Save predictions
            for i in range(images.shape[0]):
                if saved_count >= num_samples:
                    break
                
                # Convert mask to numpy: [1, H, W] -> [H, W]
                pred_mask = pred_binary[i, 0].cpu().numpy()
                
                # Convert to {0, 255} for PNG
                pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
                
                # Save as PNG
                from PIL import Image
                pred_image = Image.fromarray(pred_mask_uint8, mode='L')
                pred_image.save(output_dir / f"prediction_{saved_count}.png")
                
                saved_count += 1
    
    print(f"Saved {saved_count} prediction masks to {output_dir}")


def freeze_encoder(model: nn.Module):
    """Freeze encoder part of U-Net (optional, recommended initially)."""
    # Freeze encoder layers
    for param in model.segmentation_model.enc1.parameters():
        param.requires_grad = False
    for param in model.segmentation_model.enc2.parameters():
        param.requires_grad = False
    for param in model.segmentation_model.enc3.parameters():
        param.requires_grad = False
    for param in model.segmentation_model.enc4.parameters():
        param.requires_grad = False
    for param in model.segmentation_model.bottleneck.parameters():
        param.requires_grad = False
    
    print("Frozen encoder layers")


def cache_prompt_embeddings(
    model: nn.Module,
    prompts: Dict[str, list],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Cache CLIP text embeddings for all prompts.
    
    Args:
        model: LanguageConditionedSegmentationModel with CLIP text encoder
        prompts: Dictionary mapping class names to lists of prompt strings
        device: Device to run encoding on
    
    Returns:
        Dictionary mapping prompt strings to embedding tensors
    """
    print("Caching CLIP text embeddings for all prompts...")
    embeddings_cache = {}
    
    # Collect all unique prompts
    all_prompts = set()
    for prompt_list in prompts.values():
        all_prompts.update(prompt_list)
    
    # Also add validation prompt
    all_prompts.add("segment crack")
    
    # Encode all prompts once
    model.text_encoder.eval()
    with torch.no_grad():
        for prompt in all_prompts:
            # Use the existing CLIPTextEncoder.forward() method which handles tokenization correctly
            text_embedding = model.text_encoder(prompt)
            # Store as CPU tensor to save GPU memory, move to device when needed
            embeddings_cache[prompt] = text_embedding.cpu()
    
    print(f"Cached {len(embeddings_cache)} prompt embeddings")
    return embeddings_cache


def main():
    parser = argparse.ArgumentParser(description="Train language-conditioned crack segmentation model")
    
    # Data paths
    parser.add_argument("--data_root", type=str, default="Dataset/cracks",
                        help="Root directory containing train/valid splits")
    parser.add_argument("--prompts_path", type=str, default="Dataset/prompts.json",
                        help="Path to prompts.json file")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Model settings
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model name")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder part of U-Net")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints and outputs")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction visualizations (only after training completes)")
    
    # Validation settings
    parser.add_argument("--val_every_n_epochs", type=int, default=1,
                        help="Run validation every N epochs (default: 1)")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device detection with diagnostics
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        print("CUDA not available.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA compiled version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.prompts_path)
    print(f"Loaded prompts for classes: {list(prompts.keys())}")
    
    # Create datasets
    train_dataset = CrackSegmentationDataset(
        annotations_path=os.path.join(args.data_root, "train", "_annotations.coco.json"),
        images_dir=os.path.join(args.data_root, "train"),
        split="train"
    )
    
    val_dataset = CrackSegmentationDataset(
        annotations_path=os.path.join(args.data_root, "valid", "_annotations.coco.json"),
        images_dir=os.path.join(args.data_root, "valid"),
        split="valid"
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageConditionedSegmentationModel(clip_model_name=args.clip_model, device=device_str)
    model = model.to(device)
    
    # Use channels_last memory format for better performance on CUDA
    use_channels_last = cuda_available
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("Using channels_last memory format")
    
    # Freeze CLIP text encoder (already frozen in model definition)
    # Optionally freeze encoder
    if args.freeze_encoder:
        freeze_encoder(model)
    
    # Cache CLIP text embeddings for all prompts
    prompt_embeddings_cache = cache_prompt_embeddings(model, prompts, device)
    
    # Enable mixed precision training (AMP) for CUDA
    use_amp = cuda_available
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Loss function
    criterion = CombinedLoss()
    
    # Optimizer (only trainable parameters)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"crack_attention_unet_v1_{timestamp}"
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print(f"TensorBoard logs: {log_dir}")
    print(f"View with: tensorboard --logdir runs/")
    
    # Training loop
    best_val_dice = 0.0
    global_step = 0
    
    # Save training config
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'clip_model': args.clip_model,
        'freeze_encoder': args.freeze_encoder,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'tensorboard_run': run_name
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nStarting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Output directory: {output_dir}\n")
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            prompt_embeddings_cache, prompts, writer, global_step, 
            scaler, use_amp, class_name="crack"
        )
        
        # Validate only every N epochs
        if (epoch + 1) % args.val_every_n_epochs == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, 
                prompt_embeddings_cache, use_amp, class_name="crack"
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log validation metrics (once per epoch)
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/dice', val_metrics['dice'], epoch)
            writer.add_scalar('val/iou', val_metrics['iou'], epoch)
            
            # Log learning rate (once per epoch)
            writer.add_scalar('lr', current_lr, epoch)
            
            # Learning rate scheduling
            old_lr = current_lr
            scheduler.step(val_metrics['loss'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Console logging: one line per epoch
            print(f"Epoch {epoch + 1}/{args.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'config': config
                }, output_dir / "best_model.pth")
                print(f"  ✓ Saved best model (Dice: {best_val_dice:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'config': config
                }, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")
        else:
            # No validation this epoch, just log training metrics
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{args.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"LR: {current_lr:.2e} (validation skipped)")
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed! Best validation Dice: {best_val_dice:.4f}")
    print(f"TensorBoard logs saved to: {log_dir}")
    
    # Save predictions for a few validation samples (only after training completes)
    if args.save_predictions:
        print("\nSaving prediction visualizations...")
        save_predictions(
            model, val_loader, device,
            output_dir / "predictions",
            prompt_embeddings_cache,
            use_amp,
            num_samples=10
        )


if __name__ == "__main__":
    main()

