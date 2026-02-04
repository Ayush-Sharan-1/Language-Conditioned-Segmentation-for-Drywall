# Language-Conditioned Segmentation for Drywall

A PyTorch implementation for language-conditioned semantic segmentation, supporting both crack detection and drywall taping area segmentation. This project uses CLIP text embeddings to condition a U-Net segmentation model, allowing flexible segmentation through natural language prompts.

## Features

- **Language-conditioned segmentation**: Use text prompts to guide segmentation
- **Multi-dataset support**: Train on cracks or drywall taping areas
- **Pseudo-mask generation**: Convert bounding box annotations to segmentation masks
- **Inference**: Single image and batch processing with visualization
- **Training utilities**: Mixed precision training, checkpointing, TensorBoard logging

## Project Structure

```
.
├── Scripts/
│   ├── train.py              # Main training script
│   ├── inference.py          # Inference script (single/batch)
│   ├── generate_masks.py     # Pseudo-mask generation from bboxes
│   ├── model.py              # Model architecture
│   ├── dataset.py            # Dataset loader
│   ├── losses.py             # Loss functions
│   └── utils.py              # Utility functions
├── Dataset/
│   ├── cracks/               # Crack dataset
│   ├── drywall-post/         # Drywall dataset
│   └── prompts.json          # Text prompts configuration
└── Models/                   # Trained model checkpoints
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install opencv-python pillow numpy
pip install pycocotools tqdm
pip install tensorboard
pip install clip-by-openai
```

## Usage

### 1. Generate Pseudo Masks (`generate_masks.py`)

Convert bounding box annotations to segmentation masks for training. This script uses edge detection and color analysis to create approximate segmentation masks from bounding boxes.

#### Configuration

Create a JSON configuration file (see `Scripts/config_gen_pseudo_mask.json` for example):

```json
{
  "data_root": "Dataset/drywall-pre/train",
  "annotations_path": "Dataset/drywall-pre/train/_annotations.coco.json",
  "output_dir": "Dataset/drywall-post/",
  "num_images": null,
  "train_ratio": 0.8,
  "valid_ratio": 0.1,
  "test_ratio": 0.1,
  "edge_detector_type": "canny",
  "canny_low_threshold": 20.0,
  "canny_high_threshold": 40.0,
  "color_difference_threshold": 15.0,
  "min_edge_length": 100,
  "dilation_kernel_size": 3,
  "erosion_kernel_size": 2
}
```

#### Parameters

- **`data_root`**: Directory containing images
- **`annotations_path`**: Path to COCO format annotations JSON
- **`output_dir`**: Output directory for generated splits
- **`num_images`**: Number of images to process (null for all)
- **`train_ratio`**, **`valid_ratio`**, **`test_ratio`**: Dataset split ratios
- **`edge_detector_type`**: "canny" or "sobel"
- **`canny_low_threshold`**, **`canny_high_threshold`**: Canny edge detector thresholds
- **`color_difference_threshold`**: LAB color space threshold for filtering edges
- **`min_edge_length`**: Minimum edge length in pixels
- **`dilation_kernel_size`**, **`erosion_kernel_size`**: Morphological operation parameters

#### Usage

```bash
python Scripts/generate_masks.py Scripts/config_gen_pseudo_mask.json
```

#### Output

The script generates:
- Train/valid/test splits with images and `_annotations.coco.json` files
- Visualization overlays showing bounding boxes and generated masks
- COCO format segmentation annotations

### 2. Training (`train.py`)

Train a language-conditioned segmentation model on cracks or drywall datasets.

#### Basic Usage

**For cracks dataset:**
```bash
python Scripts/train.py --dataset_type cracks
```

**For drywall dataset:**
```bash
python Scripts/train.py --dataset_type drywall
```

#### Arguments

**Dataset Selection:**
- `--dataset_type`: Choose `"cracks"` or `"drywall"` (default: `"cracks"`)
- `--data_root`: Root directory containing train/valid splits (auto-set if not provided)
- `--prompts_path`: Path to prompts.json file (default: `"Dataset/prompts.json"`)

**Training Hyperparameters:**
- `--batch_size`: Batch size (default: 4)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--seed`: Random seed (default: 42)

**Model Settings:**
- `--clip_model`: CLIP model name (default: `"ViT-B/32"`)
- `--freeze_encoder`: Freeze U-Net encoder layers

**Output Settings:**
- `--output_dir`: Directory to save checkpoints (default: `"checkpoints"`)
- `--save_predictions`: Save prediction visualizations after training
- `--val_every_n_epochs`: Run validation every N epochs (default: 1)

**Resume Training:**
- `--resume`: Path to checkpoint to resume from

#### Example

```bash
python Scripts/train.py \
    --dataset_type cracks \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --output_dir checkpoints/cracks_model \
    --save_predictions
```

#### Output

- **Checkpoints**: Saved in `output_dir/`
  - `best_model.pth`: Best model based on validation Dice score
  - `checkpoint_epoch_N.pth`: Periodic checkpoints every 10 epochs
  - `config.json`: Training configuration
- **TensorBoard logs**: Saved in `runs/` directory
  - View with: `tensorboard --logdir runs/`

### 3. Inference (`inference.py`)

Run inference on single images or batches with customizable text prompts.

#### Single Image Mode

```bash
python Scripts/inference.py \
    --image path/to/image.jpg \
    --checkpoint Models/crack_only.pth \
    --prompt "segment cracks"
```

**Optional arguments:**
- `--output`: Custom output path for mask (default: `{imageID}_{prompt}_mask.png`)
- `--visualization`: Custom path for visualization (default: auto-generated)
- `--threshold`: Binary mask threshold (default: 0.5)
- `--device`: Device to use: `auto`, `cuda`, or `cpu` (default: `auto`)

#### Batch Mode

```bash
python Scripts/inference.py \
    --folder Dataset/cracks/test \
    --checkpoint Models/crack_only.pth \
    --output_dir Dataset/cracks/inf_results \
    --prompt "segment cracks" \
    --save_visualizations \
    --num_images 50
```

**Arguments:**
- `--folder`: Folder containing images to process
- `--checkpoint`: Path to model checkpoint
- `--output_dir`: Output directory for results (default: `{folder}/predictions`)
- `--prompt`: Text prompt for segmentation
- `--save_visualizations`: Save visualization overlays
- `--num_images`: Limit number of images to process (default: all)
- `--threshold`: Binary mask threshold (default: 0.5)

#### Output File Naming

Output files follow the format: `{imageID}_{sanitized_prompt}_{type}.png`

- **Mask files**: `image001_segment_cracks_mask.png`
- **Visualization files**: `image001_segment_cracks_overlay.png`

#### Visualization

When `--save_visualizations` is used, the script creates side-by-side visualizations:

- **With ground truth**: `Original | Ground Truth | Predicted` (3 panels)

## Dataset Format

### Expected Structure

```
Dataset/
├── cracks/
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── *.jpg
│   └── valid/
│       ├── _annotations.coco.json
│       └── *.jpg
└── drywall-post/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── valid/
        ├── _annotations.coco.json
        └── *.jpg
```

### Prompts Configuration

The `Dataset/prompts.json` file defines text prompts for each class:

```json
{
  "crack": [
    "segment crack",
    "segment cracks",
    "find crack",
    "detect crack region"
  ],
  "taping_area": [
    "segment taping area",
    "segment drywall tape",
    "find drywall seam"
  ]
}
```
