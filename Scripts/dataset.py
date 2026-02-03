"""
Dataset class for crack segmentation with COCO format annotations.
"""

import json
import os
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_utils


class CrackSegmentationDataset(Dataset):
    """
    Dataset for crack segmentation from COCO format annotations.
    
    Returns:
        - image: torch.Tensor, shape [3, H, W], normalized to [0, 1]
        - mask: torch.Tensor, shape [1, H, W], binary mask with values {0, 1}
        - class_label: str, always "crack"
    """
    
    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        transform=None,
        split: str = "train"
    ):
        """
        Args:
            annotations_path: Path to COCO format annotations JSON file
            images_dir: Directory containing images
            transform: Optional torchvision transforms
            split: Dataset split name ("train", "valid", "test")
        """
        self.annotations_path = annotations_path
        self.images_dir = images_dir
        self.transform = transform
        self.split = split
        
        # Load COCO annotations
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image and annotation mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
        
        # Create list of valid image IDs (images with annotations)
        self.image_ids = [
            img_id for img_id in self.images.keys()
            if img_id in self.image_annotations
        ]
        
        print(f"Loaded {len(self.image_ids)} images with annotations from {split} split")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _polygon_to_mask(self, polygon: List[List[float]], height: int, width: int) -> np.ndarray:
        """Convert polygon annotation to binary mask."""
        # Flatten polygon coordinates
        if isinstance(polygon[0], list):
            # Multiple polygons or single polygon
            if len(polygon) == 1:
                coords = np.array(polygon[0], dtype=np.float32).reshape(-1, 2)
            else:
                # Multiple polygons - combine them
                mask = np.zeros((height, width), dtype=np.uint8)
                for poly in polygon:
                    coords = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    # Create mask for this polygon
                    from PIL import Image, ImageDraw
                    img = Image.new('L', (width, height), 0)
                    draw = ImageDraw.Draw(img)
                    draw.polygon([tuple(p) for p in coords], fill=1)
                    poly_mask = np.array(img)
                    mask = np.maximum(mask, poly_mask)
                return mask.astype(np.float32)
        else:
            coords = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        
        # Create mask from polygon
        from PIL import Image, ImageDraw
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in coords], fill=1)
        mask = np.array(img, dtype=np.float32)
        
        return mask
    
    def _rle_to_mask(self, rle: Dict, height: int, width: int) -> np.ndarray:
        """Convert RLE annotation to binary mask."""
        if isinstance(rle, list):
            # RLE format: [size, counts]
            rle_obj = {'size': [height, width], 'counts': rle}
        else:
            rle_obj = rle
        
        mask = mask_utils.decode(rle_obj)
        return mask.astype(np.float32)
    
    def _annotations_to_mask(
        self,
        annotations: List[Dict],
        height: int,
        width: int
    ) -> np.ndarray:
        """Convert multiple annotations to a single binary mask."""
        mask = np.zeros((height, width), dtype=np.float32)
        
        for ann in annotations:
            seg = ann['segmentation']
            
            if isinstance(seg, dict):
                # RLE format
                ann_mask = self._rle_to_mask(seg, height, width)
            elif isinstance(seg, list):
                # Polygon format
                ann_mask = self._polygon_to_mask(seg, height, width)
            else:
                continue
            
            # Combine masks (union)
            mask = np.maximum(mask, ann_mask)
        
        # Ensure binary: values are exactly 0 or 1
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image: [3, H, W] tensor, normalized to [0, 1]
            mask: [1, H, W] tensor, binary mask with values {0, 1}
            class_label: str, "crack"
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        annotations = self.image_annotations[image_id]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Get image dimensions
        height = image_info['height']
        width = image_info['width']
        
        # Convert annotations to binary mask
        mask = self._annotations_to_mask(annotations, height, width)
        
        # Convert to tensors
        # Image: [H, W, 3] -> [3, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        # Mask: [H, W] -> [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Apply transforms if provided
        if self.transform:
            # Apply transform to both image and mask
            # Note: transforms should handle both tensors
            image = self.transform(image)
            # For mask, we might need special handling depending on transform
        
        # Ensure mask is binary {0, 1}
        mask = (mask > 0.5).float()
        
        return image, mask, "crack"

