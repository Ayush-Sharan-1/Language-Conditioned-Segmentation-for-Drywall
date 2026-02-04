"""
Training script for language-conditioned segmentation supporting cracks and drywall taping areas.
Supports negative prompt augmentation for improved prompt adherence.
"""

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, BatchSampler, Sampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image
from collections import defaultdict

from dataset import CrackSegmentationDataset
from model import LanguageConditionedSegmentationModel
from losses import CombinedLoss, DiceLoss
from utils import load_prompts, sample_prompt


class SimpleImageMaskDataset(Dataset):
    """
    Dataset for simple image/mask pairs (for pseudo masks).
    
    Assumes directory structure:
        data_root/
            images/
            masks/
    
    Returns:
        - image: torch.Tensor, shape [3, H, W], normalized to [0, 1]
        - mask: torch.Tensor, shape [1, H, W], binary mask with values {0, 1}
        - class_label: str, class name (e.g., "crack" or "taping_area")
    """
    
    def __init__(
        self,
        data_root: str,
        class_name: str,
        class_id: int,
        transform=None,
        split: str = "train"
    ):
        """
        Args:
            data_root: Root directory containing images/ and masks/ subdirectories
            class_name: Name of the class (e.g., "crack" or "taping_area")
            class_id: Class ID for multi-class mapping (1 for crack, 2 for taping_area)
            transform: Optional torchvision transforms
            split: Dataset split name ("train", "valid", "test")
        """
        self.data_root = Path(data_root)
        self.class_name = class_name
        self.class_id = class_id
        self.transform = transform
        self.split = split
        
        # Try multiple possible directory structures
        # 1. data_root/split/images and data_root/split/masks (preferred)
        # 2. data_root/images and data_root/masks (fallback)
        
        if (self.data_root / split / "images").exists() and (self.data_root / split / "masks").exists():
            images_dir = self.data_root / split / "images"
            masks_dir = self.data_root / split / "masks"
        elif (self.data_root / "images").exists() and (self.data_root / "masks").exists():
            images_dir = self.data_root / "images"
            masks_dir = self.data_root / "masks"
        else:
            raise ValueError(
                f"Could not find images/ and masks/ directories. "
                f"Tried: {self.data_root / split / 'images'} and {self.data_root / split / 'masks'}, "
                f"or {self.data_root / 'images'} and {self.data_root / 'masks'}"
            )
        
        # Get list of image files
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # Match with mask files
        self.samples = []
        for img_file in image_files:
            # Try to find corresponding mask file
            mask_name = img_file.stem + ".png"  # Assume masks are PNG
            mask_file = masks_dir / mask_name
            
            if mask_file.exists():
                self.samples.append((img_file, mask_file))
            else:
                # Try other extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    mask_file = masks_dir / (img_file.stem + ext)
                    if mask_file.exists():
                        self.samples.append((img_file, mask_file))
                        break
        
        if len(self.samples) == 0:
            raise ValueError(f"No matching image/mask pairs found in {images_dir} and {masks_dir}")
        
        print(f"Loaded {len(self.samples)} image/mask pairs for {class_name} ({split} split)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image: [3, H, W] tensor, normalized to [0, 1]
            mask: [1, H, W] tensor, binary mask with values {0, 1}
            class_label: str, class name
        """
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Ensure mask is binary {0, 1}
        mask = (mask > 0.5).astype(np.float32)
        
        # Convert to tensors
        # Image: [H, W, 3] -> [3, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        # Mask: [H, W] -> [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Ensure mask is binary {0, 1}
        mask = (mask > 0.5).float()
        
        return image, mask, self.class_name


class DatasetWithNegativeAugmentation(Dataset):
    """
    Wrapper dataset that applies negative prompt augmentation.
    
    With probability negative_prompt_prob:
        - Replaces the correct prompt with a prompt from another class
        - Replaces the target mask with all background (zeros)
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        class_name: str,
        other_class_name: Optional[str],
        prompts: Dict[str, List[str]],
        negative_prompt_prob: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Args:
            base_dataset: Base dataset to wrap
            class_name: Name of the class for this dataset
            other_class_name: Name of the other class (for negative prompts). If None, uses prompts from other available classes.
            prompts: Dictionary mapping class names to prompt lists
            negative_prompt_prob: Probability of applying negative prompt augmentation
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.class_name = class_name
        self.other_class_name = other_class_name
        self.prompts = prompts
        self.negative_prompt_prob = negative_prompt_prob
        
        # Set up random number generator for this dataset
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
        
        # Get available class names from prompts
        self.available_classes = list(prompts.keys())
        if self.other_class_name is None and len(self.available_classes) > 1:
            # Use the other class (not the current one)
            other_classes = [c for c in self.available_classes if c != self.class_name]
            if other_classes:
                self.other_class_name = other_classes[0]
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image: [3, H, W] tensor
            mask: [1, H, W] tensor, binary mask (may be all zeros for negative samples)
            class_label: str, class name (may be different from dataset class for negative samples)
        """
        image, mask, class_label = self.base_dataset[idx]
        
        # Determine if we should apply negative augmentation
        is_negative = self.rng.random() < self.negative_prompt_prob
        
        if is_negative:
            # Use prompt from other class
            if self.other_class_name and self.other_class_name in self.prompts:
                prompt_class = self.other_class_name
            else:
                # Fallback: use a random other class or same class if only one available
                other_classes = [c for c in self.available_classes if c != self.class_name]
                if other_classes:
                    prompt_class = self.rng.choice(other_classes)
                else:
                    prompt_class = self.class_name  # Fallback to same class if no other available
            
            # Replace mask with all zeros (background)
            mask = torch.zeros_like(mask)
            class_label = prompt_class
        
        return image, mask, class_label


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class ClassGroupedBatchSampler(BatchSampler):
    """
    Batch sampler that groups samples by class, ensuring each batch contains only one class.
    This allows using a single prompt per batch for better performance.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            dataset: Dataset that returns (image, mask, class_label)
            batch_size: Batch size
            shuffle: Whether to shuffle batches (not individual samples)
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        print("Grouping samples by class for efficient batching...")
        for idx in range(len(dataset)):
            # Get class label by accessing dataset
            _, _, class_label = dataset[idx]
            self.class_indices[class_label].append(idx)
        
        print(f"  Found classes: {list(self.class_indices.keys())}")
        for class_name, indices in self.class_indices.items():
            print(f"    {class_name}: {len(indices)} samples")
        
        # Create batches for each class
        self.batches = []
        self.batch_classes = []  # Track which class each batch belongs to
        for class_name, indices in self.class_indices.items():
            # Shuffle indices within each class
            if shuffle and seed is not None:
                rng = random.Random(seed + hash(class_name))
                rng.shuffle(indices)
            elif shuffle:
                random.shuffle(indices)
            
            # Create batches from this class
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                self.batches.append(batch)
                self.batch_classes.append(class_name)
        
        # Shuffle batches across classes
        if shuffle:
            if seed is not None:
                random.seed(seed)
            # Shuffle batches and their corresponding class labels together
            combined = list(zip(self.batches, self.batch_classes))
            random.shuffle(combined)
            self.batches, self.batch_classes = zip(*combined)
            self.batches = list(self.batches)
            self.batch_classes = list(self.batch_classes)
        
        print(f"  Created {len(self.batches)} batches (grouped by class)")
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


def compute_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    class_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute Dice score and IoU for binary segmentation.
    
    Args:
        predictions: Logits [B, 1, H, W]
        targets: Binary masks [B, 1, H, W] with values {0, 1}
        class_mask: Optional boolean mask [B] indicating which samples belong to this class
    
    Returns:
        Dictionary with 'dice' and 'iou' scores
    """
    # Apply sigmoid and threshold
    pred_probs = torch.sigmoid(predictions)
    pred_binary = (pred_probs > 0.5).float()
    
    # Filter by class if class_mask is provided
    if class_mask is not None:
        if class_mask.sum() == 0:
            # No samples of this class in batch
            return {'dice': 0.0, 'iou': 0.0}
        pred_binary = pred_binary[class_mask]
        targets = targets[class_mask]
    
    if pred_binary.numel() == 0:
        return {'dice': 0.0, 'iou': 0.0}
    
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
    negative_prompt_prob: float,
    available_classes: List[str]
) -> Tuple[Dict[str, float], int]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    num_batches = 0
    
    # Track metrics per class
    class_metrics = defaultdict(lambda: {'dice': 0.0, 'count': 0})
    negative_count = 0
    positive_count = 0
    
    # Individual loss components for logging
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    
    # Get batch sampler to access class information
    batch_sampler = getattr(dataloader, 'batch_sampler', None)
    has_class_grouping = isinstance(batch_sampler, ClassGroupedBatchSampler)
    
    for batch_idx, batch_data in enumerate(dataloader):
        # All datasets return (image, mask, class_label)
        images, masks, class_labels = batch_data
        
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Determine negative samples by checking if mask is all zeros
        # This is approximate but works for our use case
        batch_size = images.shape[0]
        is_negative_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if masks[i].sum() < 1e-6:  # Mask is essentially all zeros
                is_negative_flags[i] = True
        
        # Convert to channels_last memory format for better performance on CUDA
        if use_amp:
            images = images.to(memory_format=torch.channels_last)
            masks = masks.to(memory_format=torch.channels_last)
        
        # With class-grouped batching, all samples in batch have the same class (except negative samples)
        # So we can use a single prompt for the batch - much faster!
        if has_class_grouping:
            # Get the class name for this batch from the sampler
            batch_class_name = batch_sampler.batch_classes[batch_idx]
            # Sample one prompt for the entire batch
            prompt = sample_prompt(batch_class_name, prompts, is_training=True)
            text_embedding = prompt_embeddings_cache[prompt]
            if text_embedding.shape[0] == 1:
                text_embedding = text_embedding.squeeze(0)  # [1, text_dim] -> [text_dim]
            text_embedding = text_embedding.to(device)  # Single GPU transfer per batch!
            # Expand to batch size (model will handle this, but we can do it explicitly)
            text_embedding_batch = text_embedding.unsqueeze(0).expand(batch_size, -1)
            
            # Count positive/negative samples
            positive_count += (~is_negative_flags).sum().item()
            negative_count += is_negative_flags.sum().item()
        else:
            # Fallback: handle mixed batches (shouldn't happen with ClassGroupedBatchSampler)
            prompts_list = []
            for i in range(batch_size):
                class_label = class_labels[i] if isinstance(class_labels, (list, tuple)) else class_labels
                is_negative = is_negative_flags[i].item()
                
                if is_negative:
                    negative_count += 1
                else:
                    positive_count += 1
                
                prompt = sample_prompt(class_label, prompts, is_training=True)
                prompts_list.append(prompt)
            
            # Get unique prompts and transfer to device in one batch
            unique_prompts = list(set(prompts_list))
            unique_embeddings = {}
            for prompt in unique_prompts:
                emb = prompt_embeddings_cache[prompt]
                if emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                unique_embeddings[prompt] = emb.to(device)
            
            text_embeddings = [unique_embeddings[prompt] for prompt in prompts_list]
            text_embedding_batch = torch.stack(text_embeddings, dim=0)
        
        # Forward pass
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model.forward_with_embedding(images, text_embedding_batch)
                loss = criterion(logits, masks)
                # Compute individual loss components for logging
                bce_loss = bce_loss_fn(logits, masks)
                dice_loss = dice_loss_fn(logits, masks)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model.forward_with_embedding(images, text_embedding_batch)
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
        
        # Compute per-class metrics (only for positive samples)
        positive_mask = ~is_negative_flags
        if positive_mask.any():
            for class_name in available_classes:
                # Find samples of this class
                class_mask = torch.zeros(batch_size, dtype=torch.bool)
                for i in range(batch_size):
                    if positive_mask[i]:
                        sample_class = class_labels[i] if isinstance(class_labels, (list, tuple)) else class_labels
                        if sample_class == class_name:
                            class_mask[i] = True
                
                if class_mask.any():
                    metrics = compute_metrics(logits, masks, class_mask)
                    class_metrics[class_name]['dice'] += metrics['dice']
                    class_metrics[class_name]['count'] += 1
        
        # Log every 10 iterations
        if (batch_idx + 1) % 10 == 0:
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            writer.add_scalar('train/bce_loss', bce_loss.item(), global_step)
            writer.add_scalar('train/dice_loss', dice_loss.item(), global_step)
            global_step += 1
            
            # Compute per-class Dice for this batch (only for positive samples)
            batch_dice_str = ""
            positive_mask = ~is_negative_flags
            if positive_mask.any():
                for class_name in available_classes:
                    # Find samples of this class in current batch
                    class_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for i in range(batch_size):
                        if positive_mask[i]:
                            sample_class = class_labels[i] if isinstance(class_labels, (list, tuple)) else class_labels
                            if sample_class == class_name:
                                class_mask[i] = True
                    
                    if class_mask.any():
                        metrics = compute_metrics(logits, masks, class_mask)
                        batch_dice_str += f", {class_name} Dice: {metrics['dice']:.4f}"
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[{timestamp}] Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}{batch_dice_str}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_bce_loss = total_bce_loss / num_batches if num_batches > 0 else 0.0
    avg_dice_loss = total_dice_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute average per-class metrics
    metrics_dict = {
        'loss': avg_loss,
        'bce_loss': avg_bce_loss,
        'dice_loss': avg_dice_loss,
        'negative_samples': negative_count,
        'positive_samples': positive_count
    }
    
    for class_name in available_classes:
        if class_metrics[class_name]['count'] > 0:
            avg_dice = class_metrics[class_name]['dice'] / class_metrics[class_name]['count']
            metrics_dict[f'{class_name}_dice'] = avg_dice
    
    return metrics_dict, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prompt_embeddings_cache: Dict[str, torch.Tensor],
    prompts: Dict[str, List[str]],
    use_amp: bool,
    available_classes: List[str]
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Track metrics per class
    class_metrics = defaultdict(lambda: {'dice': 0.0, 'iou': 0.0, 'count': 0})
    
    # Use fixed prompts for validation
    val_prompts = {}
    for class_name in available_classes:
        # Try to find a prompt in cache that matches this class
        found = False
        for prompt in prompt_embeddings_cache.keys():
            # Check if prompt is for this class
            if class_name in prompts and prompt in prompts[class_name]:
                val_prompts[class_name] = prompt
                found = True
                break
        
        if not found:
            # Fallback: use first prompt from class list or default
            if class_name in prompts and len(prompts[class_name]) > 0:
                val_prompts[class_name] = prompts[class_name][0]
            else:
                val_prompts[class_name] = f"segment {class_name}"
    
    with torch.no_grad():
        for batch_data in dataloader:
            # All datasets return (image, mask, class_label)
            images, masks, class_labels = batch_data
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Convert to channels_last memory format for better performance on CUDA
            if use_amp:
                images = images.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)
            
            # Use fixed prompts for validation
            batch_size = images.shape[0]
            prompts_list = []
            
            for i in range(batch_size):
                class_label = class_labels[i] if isinstance(class_labels, (list, tuple)) else class_labels
                # Use validation prompt for this class
                val_prompt = val_prompts.get(class_label, list(val_prompts.values())[0])
                prompts_list.append(val_prompt)
            
            # Get unique prompts and transfer to device in one batch (optimization)
            unique_prompts = list(set(prompts_list))
            unique_embeddings = {}
            for prompt in unique_prompts:
                emb = prompt_embeddings_cache[prompt]
                if emb.shape[0] == 1:
                    emb = emb.squeeze(0)  # [1, text_dim] -> [text_dim]
                unique_embeddings[prompt] = emb.to(device)  # Transfer once per unique prompt
            
            # Map embeddings to samples and stack
            text_embeddings = [unique_embeddings[prompt] for prompt in prompts_list]
            text_embedding_batch = torch.stack(text_embeddings, dim=0)
            
            # Forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model.forward_with_embedding(images, text_embedding_batch)
                    loss = criterion(logits, masks)
            else:
                logits = model.forward_with_embedding(images, text_embedding_batch)
                loss = criterion(logits, masks)
            
            # Compute per-class metrics
            for class_name in available_classes:
                # Find samples of this class
                class_mask = torch.zeros(batch_size, dtype=torch.bool)
                for i in range(batch_size):
                    sample_class = class_labels[i] if isinstance(class_labels, (list, tuple)) else class_labels
                    if sample_class == class_name:
                        class_mask[i] = True
                
                if class_mask.any():
                    metrics = compute_metrics(logits, masks, class_mask)
                    class_metrics[class_name]['dice'] += metrics['dice']
                    class_metrics[class_name]['iou'] += metrics['iou']
                    class_metrics[class_name]['count'] += 1
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics_dict = {'loss': avg_loss}
    
    # Compute average per-class metrics
    for class_name in available_classes:
        if class_metrics[class_name]['count'] > 0:
            avg_dice = class_metrics[class_name]['dice'] / class_metrics[class_name]['count']
            avg_iou = class_metrics[class_name]['iou'] / class_metrics[class_name]['count']
            metrics_dict[f'{class_name}_dice'] = avg_dice
            metrics_dict[f'{class_name}_iou'] = avg_iou
    
    return metrics_dict


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
    
    # Also add default validation prompts
    for class_name in prompts.keys():
        all_prompts.add(f"segment {class_name}")
    
    # Encode all prompts once
    model.text_encoder.eval()
    with torch.no_grad():
        for prompt in all_prompts:
            text_embedding = model.text_encoder(prompt)
            # Store as CPU tensor to save GPU memory, move to device when needed
            embeddings_cache[prompt] = text_embedding.cpu()
    
    print(f"Cached {len(embeddings_cache)} prompt embeddings")
    return embeddings_cache


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


def create_dataset(
    data_root: str,
    class_name: str,
    class_id: int,
    prompts: Dict[str, List[str]],
    negative_prompt_prob: float,
    split: str = "train",
    seed: Optional[int] = None,
    use_coco_format: bool = False
) -> Dataset:
    """
    Create a dataset for the given class.
    
    Args:
        data_root: Root directory of the dataset
        class_name: Name of the class (e.g., "crack" or "taping_area")
        class_id: Class ID (1 for crack, 2 for taping_area)
        prompts: Dictionary of prompts
        negative_prompt_prob: Probability of negative prompt augmentation
        split: Dataset split ("train" or "valid")
        seed: Random seed for reproducibility
        use_coco_format: If True, use COCO format dataset; if False, use simple image/mask pairs
    
    Returns:
        Dataset instance
    """
    # Auto-detect COCO format by checking for _annotations.coco.json
    data_root_path = Path(data_root)
    
    # Check if data_root already includes the split name
    if str(data_root_path).endswith(split):
        # data_root is already the split directory (e.g., Dataset/cracks/train)
        annotations_path = data_root_path / "_annotations.coco.json"
        images_dir = str(data_root_path)
        is_split_in_path = True
    else:
        # data_root is parent, append split
        annotations_path = data_root_path / split / "_annotations.coco.json"
        images_dir = str(data_root_path / split)
        is_split_in_path = False
    
    # Auto-detect COCO format if not explicitly set
    if not use_coco_format and annotations_path.exists():
        use_coco_format = True
        print(f"  Auto-detected COCO format (found {annotations_path})")
    
    if use_coco_format:
        # Use COCO format dataset
        if not annotations_path.exists():
            raise ValueError(f"COCO annotations not found: {annotations_path}")
        
        base_dataset = CrackSegmentationDataset(
            annotations_path=str(annotations_path),
            images_dir=images_dir,
            split=split
        )
    else:
        # Use simple image/mask pairs
        base_dataset = SimpleImageMaskDataset(
            data_root=data_root,
            class_name=class_name,
            class_id=class_id,
            split=split
        )
    
    # Wrap with negative augmentation for training
    if split == "train" and negative_prompt_prob > 0:
        # Determine other class name
        other_class_name = None
        available_classes = list(prompts.keys())
        if len(available_classes) > 1:
            other_classes = [c for c in available_classes if c != class_name]
            if other_classes:
                other_class_name = other_classes[0]
        
        # Create seed for this dataset based on class name and split
        dataset_seed = seed
        if dataset_seed is not None:
            # Make seed unique per dataset
            dataset_seed = hash(f"{class_name}_{split}_{seed}") % (2**31)
        
        dataset = DatasetWithNegativeAugmentation(
            base_dataset=base_dataset,
            class_name=class_name,
            other_class_name=other_class_name,
            prompts=prompts,
            negative_prompt_prob=negative_prompt_prob,
            seed=dataset_seed
        )
    else:
        # No negative augmentation for validation
        dataset = base_dataset
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train language-conditioned segmentation model for cracks and drywall")
    
    # Data paths
    parser.add_argument("--cracks_data_root", type=str, default=None,
                        help="Root directory for cracks dataset (optional)")
    parser.add_argument("--drywall_data_root", type=str, default=None,
                        help="Root directory for drywall taping dataset (optional)")
    parser.add_argument("--prompts_path", type=str, default="Dataset/prompts.json",
                        help="Path to prompts.json file")
    
    # Negative prompt augmentation
    parser.add_argument("--negative_prompt_prob", type=float, default=0.2,
                        help="Probability of applying negative prompt augmentation (default: 0.2)")
    
    # Dataset format
    parser.add_argument("--cracks_use_coco", action="store_true",
                        help="Use COCO format for cracks dataset (default: simple image/mask pairs)")
    parser.add_argument("--drywall_use_coco", action="store_true",
                        help="Use COCO format for drywall dataset (default: simple image/mask pairs)")
    
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
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Validate that at least one dataset is provided
    if not args.cracks_data_root and not args.drywall_data_root:
        raise ValueError("At least one of --cracks_data_root or --drywall_data_root must be specified.")
    
    # Set random seed
    set_seed(args.seed)
    
    # Device detection
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.prompts_path)
    print(f"Loaded prompts for classes: {list(prompts.keys())}")
    
    # Determine available classes based on provided datasets
    available_classes = []
    if args.cracks_data_root:
        available_classes.append("crack")
    if args.drywall_data_root:
        available_classes.append("taping_area")
    
    print(f"Training on classes: {available_classes}")
    
    # Warn if negative prompt augmentation won't work with only one class
    if args.negative_prompt_prob > 0 and len(available_classes) == 1:
        print(f"Warning: Negative prompt augmentation (prob={args.negative_prompt_prob}) requires multiple classes.")
        print("  With only one class available, negative augmentation will be limited.")
        print("  Consider providing both datasets or setting --negative_prompt_prob to 0.")
    
    # Create training datasets
    train_datasets = []
    
    if args.cracks_data_root:
        print(f"\nLoading cracks dataset from: {args.cracks_data_root}")
        try:
            cracks_train = create_dataset(
                data_root=args.cracks_data_root,
                class_name="crack",
                class_id=1,
                prompts=prompts,
                negative_prompt_prob=args.negative_prompt_prob,
                split="train",
                seed=args.seed,
                use_coco_format=args.cracks_use_coco
            )
            train_datasets.append(cracks_train)
        except Exception as e:
            print(f"Warning: Could not load cracks training dataset: {e}")
            print("Continuing without cracks dataset...")
    
    if args.drywall_data_root:
        print(f"\nLoading drywall dataset from: {args.drywall_data_root}")
        try:
            drywall_train = create_dataset(
                data_root=args.drywall_data_root,
                class_name="taping_area",
                class_id=2,
                prompts=prompts,
                negative_prompt_prob=args.negative_prompt_prob,
                split="train",
                seed=args.seed,
                use_coco_format=args.drywall_use_coco
            )
            train_datasets.append(drywall_train)
        except Exception as e:
            print(f"Warning: Could not load drywall training dataset: {e}")
            print("Continuing without drywall dataset...")
    
    if len(train_datasets) == 0:
        raise ValueError("Could not load any training datasets. Please check your data paths.")
    
    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = train_datasets[0]
    
    # Create validation datasets
    val_datasets = []
    
    if args.cracks_data_root:
        try:
            cracks_val = create_dataset(
                data_root=args.cracks_data_root,
                class_name="crack",
                class_id=1,
                prompts=prompts,
                negative_prompt_prob=0.0,  # No negative augmentation for validation
                split="valid",
                seed=args.seed,
                use_coco_format=args.cracks_use_coco
            )
            val_datasets.append(cracks_val)
        except Exception as e:
            print(f"Warning: Could not load cracks validation dataset: {e}")
    
    if args.drywall_data_root:
        try:
            # For validation, also try to auto-detect COCO format
            drywall_val = create_dataset(
                data_root=args.drywall_data_root,
                class_name="taping_area",
                class_id=2,
                prompts=prompts,
                negative_prompt_prob=0.0,  # No negative augmentation for validation
                split="valid",
                seed=args.seed,
                use_coco_format=args.drywall_use_coco
            )
            val_datasets.append(drywall_val)
        except Exception as e:
            print(f"Warning: Could not load drywall validation dataset: {e}")
            # If validation doesn't exist, that's okay - we'll just use cracks validation
    
    if len(val_datasets) == 0:
        print("Warning: No validation datasets loaded. Validation will be skipped.")
        val_dataset = None
    elif len(val_datasets) > 1:
        val_dataset = ConcatDataset(val_datasets)
    else:
        val_dataset = val_datasets[0]
    
    # Create dataloaders with class-grouped batching for better performance
    # This ensures each batch contains only one class, allowing single prompt per batch
    train_batch_sampler = ClassGroupedBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = None
    if val_dataset is not None:
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
    run_name = f"multi_class_segmentation_{timestamp}"
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print(f"TensorBoard logs: {log_dir}")
    print(f"View with: tensorboard --logdir runs/")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_dice = 0.0
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"\nResuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("  ✓ Model state loaded")
            except Exception as e:
                print(f"  ✗ Error loading model state: {e}")
                print("  Starting training from scratch...")
                args.resume = None
            
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("  ✓ Optimizer state loaded")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load optimizer state: {e}")
                
                start_epoch = checkpoint.get('epoch', 0)
                if start_epoch > 0:
                    start_epoch -= 1
                
                best_val_dice = checkpoint.get('val_dice', 0.0)
                print(f"  Resumed from epoch {start_epoch + 1}")
                print(f"  Best validation Dice so far: {best_val_dice:.4f}")
        else:
            print(f"Warning: Checkpoint file not found: {args.resume}")
    
    # Training loop
    global_step = 0
    
    # Save training config
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'clip_model': args.clip_model,
        'freeze_encoder': args.freeze_encoder,
        'negative_prompt_prob': args.negative_prompt_prob,
        'cracks_data_root': args.cracks_data_root,
        'drywall_data_root': args.drywall_data_root,
        'available_classes': available_classes,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'tensorboard_run': run_name
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nStarting training...")
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Output directory: {output_dir}")
    print(f"Negative prompt probability: {args.negative_prompt_prob}")
    print()
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            prompt_embeddings_cache, prompts, writer, global_step, 
            scaler, use_amp, args.negative_prompt_prob, available_classes
        )
        
        # Validate only every N epochs
        if val_loader is not None and (epoch + 1) % args.val_every_n_epochs == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, 
                prompt_embeddings_cache, prompts, use_amp, available_classes
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log validation metrics
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            for class_name in available_classes:
                if f'{class_name}_dice' in val_metrics:
                    writer.add_scalar(f'val/{class_name}_dice', val_metrics[f'{class_name}_dice'], epoch)
                    writer.add_scalar(f'val/{class_name}_iou', val_metrics[f'{class_name}_iou'], epoch)
            
            # Log learning rate
            writer.add_scalar('lr', current_lr, epoch)
            
            # Learning rate scheduling
            old_lr = current_lr
            scheduler.step(val_metrics['loss'])
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"  Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Compute overall validation dice (average of available classes)
            val_dice_values = [val_metrics.get(f'{cn}_dice', 0.0) for cn in available_classes if f'{cn}_dice' in val_metrics]
            overall_val_dice = sum(val_dice_values) / len(val_dice_values) if val_dice_values else 0.0
            
            # Console logging
            metric_str = " | ".join([f"{cn} Dice: {val_metrics.get(f'{cn}_dice', 0.0):.4f}" for cn in available_classes if f'{cn}_dice' in val_metrics])
            print(f"Epoch {epoch + 1}/{args.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"{metric_str} | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            if overall_val_dice > best_val_dice:
                best_val_dice = overall_val_dice
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': overall_val_dice,
                    'val_metrics': val_metrics,
                    'config': config
                }, output_dir / "best_model.pth")
                print(f"  ✓ Saved best model (Overall Dice: {best_val_dice:.4f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': overall_val_dice,
                    'val_metrics': val_metrics,
                    'config': config
                }, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")
        else:
            # No validation this epoch
            current_lr = optimizer.param_groups[0]['lr']
            metric_str = " | ".join([f"{cn} Dice: {train_metrics.get(f'{cn}_dice', 0.0):.4f}" for cn in available_classes if f'{cn}_dice' in train_metrics])
            print(f"Epoch {epoch + 1}/{args.num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"{metric_str} | "
                  f"LR: {current_lr:.2e} (validation skipped)")
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed! Best validation Dice: {best_val_dice:.4f}")
    print(f"TensorBoard logs saved to: {log_dir}")


if __name__ == "__main__":
    main()

