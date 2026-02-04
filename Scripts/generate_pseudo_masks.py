"""
Preprocessing pipeline to generate pseudo segmentation masks for drywall seams/joints
from bounding-box annotated images.

This script converts bounding box annotations into approximate segmentation masks
by detecting edges within bounding boxes and filtering them based on color similarity
across the edge (seams have similar colors on both sides).

Usage:
    python Scripts/generate_pseudo_masks.py <config.json>

Example:
    python Scripts/generate_pseudo_masks.py Scripts/config_pseudo_masks_example.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image


def load_coco_annotations(annotations_path: str) -> Dict:
    """Load COCO format annotations from JSON file."""
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def bbox_to_xyxy(bbox: List[float]) -> Tuple[int, int, int, int]:
    """
    Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2].
    
    Args:
        bbox: COCO format bounding box [x, y, width, height]
    
    Returns:
        Tuple of (x1, y1, x2, y2) as integers
    """
    x, y, w, h = bbox
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    return x1, y1, x2, y2


def crop_bbox_region(image: np.ndarray, bbox: List[float], 
                     padding: int = 10) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop bounding box region from image with optional padding.
    
    Args:
        image: Full image as numpy array [H, W, C]
        bbox: COCO format bounding box [x, y, width, height]
        padding: Pixels to add around the bbox
    
    Returns:
        Tuple of (cropped_image, (x_offset, y_offset)) where offsets are
        the coordinates of the top-left corner in the original image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    
    # Add padding and clip to image bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1)


def detect_edges(crop: np.ndarray, edge_detector_type: str = "canny",
                 canny_low: float = 50.0, canny_high: float = 150.0) -> np.ndarray:
    """
    Detect edges in the cropped region.
    
    Args:
        crop: Cropped image region [H, W, C]
        edge_detector_type: "canny" or "sobel"
        canny_low: Lower threshold for Canny edge detector
        canny_high: Upper threshold for Canny edge detector
    
    Returns:
        Binary edge map [H, W] with 1s at edges, 0s elsewhere
    """
    # Convert to grayscale if needed
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    
    if edge_detector_type == "canny":
        edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    elif edge_detector_type == "sobel":
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        # Threshold to get binary edges
        threshold = np.percentile(sobel_magnitude, 90)
        edges = (sobel_magnitude > threshold).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown edge detector type: {edge_detector_type}")
    
    return edges


def compute_color_difference_across_edge(
    crop: np.ndarray, 
    edges: np.ndarray,
    color_difference_threshold: float = 15.0
) -> np.ndarray:
    """
    Analyze edges by comparing color statistics on both sides using gradient-normal sampling.
    Keep edges where color difference is below threshold (likely seams).
    Suppress edges with large color changes (likely structural boundaries).
    
    Args:
        crop: Cropped image region [H, W, C] in RGB
        edges: Binary edge map [H, W]
        color_difference_threshold: L2 distance threshold in LAB color space
    
    Returns:
        Filtered binary edge map [H, W]
    """
    if edges.sum() == 0:
        return edges
    
    # Convert to LAB color space for better color difference perception
    if len(crop.shape) == 3:
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        # Convert to grayscale for gradient computation
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        # If grayscale, convert to 3-channel first
        crop_3ch = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(crop_3ch, cv2.COLOR_RGB2LAB)
        gray = crop
    
    # Compute Sobel gradients once per crop
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Get edge pixel coordinates
    edge_pixels = np.column_stack(np.where(edges > 0))
    
    if len(edge_pixels) == 0:
        return edges
    
    filtered_edges = np.zeros_like(edges)
    h, w = lab.shape[:2]
    
    # Sample distances for robustness
    sample_distances = [3, 5]
    
    # For each edge pixel, sample colors along gradient normal
    for y, x in edge_pixels:
        # Get local gradient at edge pixel
        grad_x = gx[y, x]
        grad_y = gy[y, x]
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Skip if gradient is too small (not a real edge)
        if grad_mag < 1e-6:
            continue
        
        # Compute unit normal direction (along gradient, perpendicular to edge)
        # Gradient is perpendicular to edge, so normal is along gradient direction
        normal_x = grad_x / grad_mag
        normal_y = grad_y / grad_mag
        
        # Sample colors on both sides along the normal
        colors1 = []
        colors2 = []
        
        for sample_dist in sample_distances:
            # Sample on both sides along the normal
            y1 = int(y + normal_y * sample_dist)
            x1 = int(x + normal_x * sample_dist)
            y2 = int(y - normal_y * sample_dist)
            x2 = int(x - normal_x * sample_dist)
            
            # Check bounds
            if (0 <= y1 < h and 0 <= x1 < w and 0 <= y2 < h and 0 <= x2 < w):
                colors1.append(lab[y1, x1])
                colors2.append(lab[y2, x2])
        
        if len(colors1) == 0:
            continue
        
        # Average colors on each side
        avg_color1 = np.mean(colors1, axis=0)
        avg_color2 = np.mean(colors2, axis=0)
        
        # Compute L2 distance in LAB space
        color_diff = np.linalg.norm(avg_color1.astype(float) - avg_color2.astype(float))
        
        # Keep edge if color difference is small (likely a seam)
        if color_diff <= color_difference_threshold:
            filtered_edges[y, x] = 255
    
    return filtered_edges


def filter_short_edges(edges: np.ndarray, min_edge_length: int = 10) -> np.ndarray:
    """
    Filter out short edges that are likely noise using contour arc-length filtering.
    
    Args:
        edges: Binary edge map [H, W]
        min_edge_length: Minimum edge length in pixels (arc length)
    
    Returns:
        Filtered binary edge map [H, W]
    """
    # Find contours using CHAIN_APPROX_NONE to preserve all points
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    filtered = np.zeros_like(edges)
    
    for contour in contours:
        # Compute contour arc length
        arc_length = cv2.arcLength(contour, closed=False)
        
        # Keep contour if arc length exceeds threshold
        if arc_length >= min_edge_length:
            # Draw the contour back into the binary mask
            cv2.drawContours(filtered, [contour], -1, 255, thickness=1)
    
    return filtered


def post_process_mask(mask: np.ndarray, 
                     dilation_kernel_size: int = 3,
                     erosion_kernel_size: int = 2) -> np.ndarray:
    """
    Post-process mask using orientation-neutral morphological operations to form continuous mask.
    
    Args:
        mask: Binary mask [H, W]
        dilation_kernel_size: Size of dilation kernel (used for closing)
        erosion_kernel_size: Size of erosion kernel (used for closing)
    
    Returns:
        Post-processed binary mask [H, W]
    """
    # Convert to uint8 if needed
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Use morphological closing (dilation followed by erosion) with symmetric kernel
    # This connects nearby edges while maintaining orientation neutrality
    kernel_size = max(dilation_kernel_size, erosion_kernel_size)
    if kernel_size > 0:
        # Use symmetric structuring element (ellipse) for orientation-neutral operation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        # Morphological closing: dilation followed by erosion
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask


def convert_to_full_image_coords(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    offset: Tuple[int, int]
) -> np.ndarray:
    """
    Convert cropped mask back to full image coordinates.
    
    Args:
        mask: Binary mask in crop coordinates [H_crop, W_crop]
        image_shape: Full image shape (height, width)
        offset: (x, y) offset of crop in original image
    
    Returns:
        Binary mask in full image coordinates [H, W]
    """
    full_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    x_offset, y_offset = offset
    h_crop, w_crop = mask.shape
    
    # Place mask in correct location
    y_end = min(y_offset + h_crop, image_shape[0])
    x_end = min(x_offset + w_crop, image_shape[1])
    
    full_mask[y_offset:y_end, x_offset:x_end] = mask[
        :(y_end - y_offset), :(x_end - x_offset)
    ]
    
    return full_mask


def process_single_bbox(
    image: np.ndarray,
    bbox: List[float],
    edge_detector_type: str = "canny",
    canny_low: float = 50.0,
    canny_high: float = 150.0,
    color_difference_threshold: float = 15.0,
    min_edge_length: int = 10,
    dilation_kernel_size: int = 3,
    erosion_kernel_size: int = 2
) -> np.ndarray:
    """
    Process a single bounding box to generate a segmentation mask.
    
    Args:
        image: Full image [H, W, C]
        bbox: COCO format bounding box [x, y, width, height]
        edge_detector_type: "canny" or "sobel"
        canny_low: Lower threshold for Canny
        canny_high: Upper threshold for Canny
        color_difference_threshold: Color difference threshold in LAB space
        min_edge_length: Minimum edge length in pixels
        dilation_kernel_size: Dilation kernel size
        erosion_kernel_size: Erosion kernel size
    
    Returns:
        Binary mask in full image coordinates [H, W]
    """
    # Step 1: Crop bounding box region
    crop, offset = crop_bbox_region(image, bbox, padding=10)
    
    if crop.size == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Step 2: Perform edge detection
    edges = detect_edges(crop, edge_detector_type, canny_low, canny_high)
    
    # Step 3-5: Analyze edges and filter by color difference
    filtered_edges = compute_color_difference_across_edge(
        crop, edges, color_difference_threshold
    )
    
    # Step 6: Filter short edges
    filtered_edges = filter_short_edges(filtered_edges, min_edge_length)
    
    # Step 6 (continued): Post-process with morphological operations
    mask_crop = post_process_mask(
        filtered_edges,
        dilation_kernel_size,
        erosion_kernel_size
    )
    
    # Step 7: Convert back to full-image coordinates
    mask_full = convert_to_full_image_coords(
        mask_crop,
        (image.shape[0], image.shape[1]),
        offset
    )
    
    return mask_full


def create_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: List[float],
    output_path: str
):
    """
    Create and save visualization overlay showing image, bounding box, and mask.
    
    Args:
        image: Original image [H, W, C]
        mask: Binary segmentation mask [H, W]
        bbox: COCO format bounding box [x, y, width, height]
        output_path: Path to save visualization
    """
    # Create a copy for visualization
    vis = image.copy()
    
    # Draw bounding box in blue
    x1, y1, x2, y2 = bbox_to_xyxy(bbox)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Overlay mask in red with transparency
    mask_colored = np.zeros_like(vis)
    mask_colored[mask > 0] = [255, 0, 0]  # Red
    vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
    
    # Save visualization
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def process_dataset(
    data_root: str,
    annotations_path: str,
    output_dir: str,
    num_images: Optional[int] = None,
    edge_detector_type: str = "canny",
    canny_low_threshold: float = 50.0,
    canny_high_threshold: float = 150.0,
    color_difference_threshold: float = 15.0,
    min_edge_length: int = 10,
    dilation_kernel_size: int = 3,
    erosion_kernel_size: int = 2
):
    """
    Main function to process the dataset and generate pseudo masks.
    
    Args:
        data_root: Path to dataset root directory containing images
        annotations_path: Path to COCO format annotations JSON file
        output_dir: Directory to save generated masks and visualizations
        num_images: Number of images to process (None for all)
        edge_detector_type: "canny" or "sobel"
        canny_low_threshold: Lower threshold for Canny edge detector
        canny_high_threshold: Upper threshold for Canny edge detector
        color_difference_threshold: L2 distance threshold in LAB color space
        min_edge_length: Minimum edge length in pixels
        dilation_kernel_size: Dilation kernel size for post-processing
        erosion_kernel_size: Erosion kernel size for post-processing
    """
    # Create output directories
    output_path = Path(output_dir)
    masks_dir = output_path / "masks"
    visualizations_dir = output_path / "visualizations"
    masks_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {annotations_path}...")
    coco_data = load_coco_annotations(annotations_path)
    
    # Build mappings
    images_dict = {img['id']: img for img in coco_data['images']}
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Get list of image IDs to process
    image_ids = list(image_annotations.keys())
    if num_images is not None:
        image_ids = image_ids[:num_images]
    
    print(f"Processing {len(image_ids)} images...")
    
    # Process each image
    for image_id in tqdm(image_ids, desc="Processing images"):
        img_info = images_dict[image_id]
        img_filename = img_info['file_name']
        img_path = Path(data_root) / img_filename
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize combined mask for this image
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Process each bounding box in this image
        annotations = image_annotations[image_id]
        for ann_idx, ann in enumerate(annotations):
            bbox = ann['bbox']
            
            # Generate mask for this bbox
            mask = process_single_bbox(
                image,
                bbox,
                edge_detector_type=edge_detector_type,
                canny_low=canny_low_threshold,
                canny_high=canny_high_threshold,
                color_difference_threshold=color_difference_threshold,
                min_edge_length=min_edge_length,
                dilation_kernel_size=dilation_kernel_size,
                erosion_kernel_size=erosion_kernel_size
            )
            
            # Combine masks (union)
            combined_mask = np.maximum(combined_mask, mask)
        
        # Save combined mask
        mask_filename = Path(img_filename).stem + ".png"
        mask_path = masks_dir / mask_filename
        mask_image = Image.fromarray(combined_mask, mode='L')
        mask_image.save(mask_path)
        
        # Create and save visualization
        vis_path = visualizations_dir / (Path(img_filename).stem + "_overlay.png")
        # Use first bbox for visualization (or combined)
        if annotations:
            create_visualization(
                image,
                combined_mask,
                annotations[0]['bbox'],
                str(vis_path)
            )
    
    print(f"\nProcessing complete!")
    print(f"Masks saved to: {masks_dir}")
    print(f"Visualizations saved to: {visualizations_dir}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['data_root', 'annotations_path', 'output_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Set defaults for optional fields
    defaults = {
        'num_images': None,
        'edge_detector_type': 'canny',
        'canny_low_threshold': 50.0,
        'canny_high_threshold': 150.0,
        'color_difference_threshold': 15.0,
        'min_edge_length': 10,
        'dilation_kernel_size': 3,
        'erosion_kernel_size': 2
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    # Validate edge_detector_type
    if config['edge_detector_type'] not in ['canny', 'sobel']:
        raise ValueError(f"edge_detector_type must be 'canny' or 'sobel', got: {config['edge_detector_type']}")
    
    return config


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_pseudo_masks.py <config.json>")
        print("\nExample config.json:")
        example_config = {
            "data_root": "Dataset/cracks/train",
            "annotations_path": "Dataset/cracks/train/_annotations.coco.json",
            "output_dir": "output/pseudo_masks",
            "num_images": None,
            "edge_detector_type": "canny",
            "canny_low_threshold": 50.0,
            "canny_high_threshold": 150.0,
            "color_difference_threshold": 15.0,
            "min_edge_length": 10,
            "dilation_kernel_size": 3,
            "erosion_kernel_size": 2
        }
        print(json.dumps(example_config, indent=2))
        print("\nNote: Use 'null' instead of None in JSON files (or omit the field to use default)")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Process dataset with configuration
    process_dataset(
        data_root=config['data_root'],
        annotations_path=config['annotations_path'],
        output_dir=config['output_dir'],
        num_images=config['num_images'],
        edge_detector_type=config['edge_detector_type'],
        canny_low_threshold=config['canny_low_threshold'],
        canny_high_threshold=config['canny_high_threshold'],
        color_difference_threshold=config['color_difference_threshold'],
        min_edge_length=config['min_edge_length'],
        dilation_kernel_size=config['dilation_kernel_size'],
        erosion_kernel_size=config['erosion_kernel_size']
    )


if __name__ == "__main__":
    main()

