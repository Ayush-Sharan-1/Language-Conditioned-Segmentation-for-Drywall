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
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

try:
    from skimage.filters import frangi
    FRANGI_AVAILABLE = True
except ImportError:
    FRANGI_AVAILABLE = False
    print("Warning: scikit-image not available. Frangi ridge detection will be disabled.")

try:
    import pycocotools.mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("Warning: pycocotools not available. COCO format output will be disabled.")


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


def detect_frangi_ridges(
    crop: np.ndarray,
    sigmas: List[float] = None,
    beta1: float = 0.5,
    beta2: float = 15.0,
    black_ridges: bool = True
) -> np.ndarray:
    """
    Detect ridges in the cropped region using Frangi filter.
    
    Args:
        crop: Cropped image region [H, W, C] in RGB
        sigmas: List of scales for multi-scale ridge detection (default: [1, 2, 3, 4, 5])
        beta1: Frangi parameter for line-like structures (smaller = more line-like)
        beta2: Frangi parameter to suppress blob-like responses
        black_ridges: If True, detect dark ridges (default for drywall seams)
    
    Returns:
        Frangi response map [H, W] with higher values at ridges
    """
    if not FRANGI_AVAILABLE:
        return np.zeros((crop.shape[0], crop.shape[1]), dtype=np.float64)
    
    if sigmas is None:
        sigmas = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop
    
    # Normalize to [0, 1] for skimage
    gray_normalized = gray.astype(np.float64) / 255.0
    
    # Apply Frangi filter
    frangi_response = frangi(
        gray_normalized,
        sigmas=sigmas,
        beta1=beta1,
        beta2=beta2,
        black_ridges=black_ridges
    )
    
    return frangi_response


def frangi_to_binary_mask(
    frangi_response: np.ndarray,
    threshold_percentile: float = 90.0,
    threshold_absolute: Optional[float] = None
) -> np.ndarray:
    """
    Convert Frangi response to binary mask using thresholding.
    
    Args:
        frangi_response: Frangi filter response [H, W]
        threshold_percentile: Percentile threshold (0-100) for top responses
        threshold_absolute: Absolute threshold (if provided, overrides percentile)
    
    Returns:
        Binary mask [H, W] with 1s at detected ridges
    """
    if threshold_absolute is not None:
        threshold = threshold_absolute
    else:
        # Use percentile-based threshold
        threshold = np.percentile(frangi_response, threshold_percentile)
    
    # Create binary mask
    binary_mask = (frangi_response >= threshold).astype(np.uint8) * 255
    
    return binary_mask


def filter_frangi_components(
    mask: np.ndarray,
    min_length: int = 10,
    min_aspect_ratio: float = 2.0
) -> np.ndarray:
    """
    Filter Frangi-derived components using geometric constraints.
    
    Args:
        mask: Binary mask [H, W]
        min_length: Minimum contour arc length in pixels
        min_aspect_ratio: Minimum aspect ratio (length/width) for elongated structures
    
    Returns:
        Filtered binary mask [H, W]
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    filtered = np.zeros_like(mask)
    
    for contour in contours:
        # Compute arc length
        arc_length = cv2.arcLength(contour, closed=False)
        
        if arc_length < min_length:
            continue
        
        # Compute bounding rectangle for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio (longer dimension / shorter dimension)
        if w > 0 and h > 0:
            aspect_ratio = max(w, h) / min(w, h)
        else:
            aspect_ratio = 0
        
        # Keep if elongated (aspect ratio >= min_aspect_ratio)
        if aspect_ratio >= min_aspect_ratio:
            cv2.drawContours(filtered, [contour], -1, 255, thickness=1)
    
    return filtered


def apply_frangi_fallback(
    crop: np.ndarray,
    frangi_sigmas: List[float] = None,
    frangi_beta1: float = 0.5,
    frangi_beta2: float = 15.0,
    frangi_threshold_percentile: float = 90.0,
    frangi_threshold_absolute: Optional[float] = None,
    frangi_min_length: int = 10,
    frangi_dilation_kernel: int = 2
) -> np.ndarray:
    """
    Apply Frangi ridge detection as fallback when edge detection fails.
    
    Args:
        crop: Cropped image region [H, W, C] in RGB
        frangi_sigmas: List of scales for Frangi filter
        frangi_beta1: Frangi parameter for line-like structures
        frangi_beta2: Frangi parameter to suppress blob-like responses
        frangi_threshold_percentile: Percentile threshold for Frangi response
        frangi_threshold_absolute: Absolute threshold (overrides percentile if provided)
        frangi_min_length: Minimum contour length for Frangi components
        frangi_dilation_kernel: Dilation kernel size for Frangi mask
    
    Returns:
        Binary mask [H, W] from Frangi detection
    """
    if not FRANGI_AVAILABLE:
        return np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    
    # Detect ridges using Frangi
    frangi_response = detect_frangi_ridges(
        crop,
        sigmas=frangi_sigmas,
        beta1=frangi_beta1,
        beta2=frangi_beta2,
        black_ridges=True
    )
    
    # Convert to binary mask
    mask = frangi_to_binary_mask(
        frangi_response,
        threshold_percentile=frangi_threshold_percentile,
        threshold_absolute=frangi_threshold_absolute
    )
    
    # Filter by geometric constraints
    mask = filter_frangi_components(mask, min_length=frangi_min_length)
    
    # Apply small dilation to create usable mask width
    if frangi_dilation_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (frangi_dilation_kernel, frangi_dilation_kernel)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    
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
    erosion_kernel_size: int = 2,
    enable_frangi: bool = True,
    frangi_sigmas: List[float] = None,
    frangi_beta1: float = 0.5,
    frangi_beta2: float = 15.0,
    frangi_threshold_percentile: float = 90.0,
    frangi_threshold_absolute: Optional[float] = None,
    frangi_min_length: int = 10,
    frangi_dilation_kernel: int = 2,
    edge_pixel_threshold: int = 10
) -> np.ndarray:
    """
    Process a single bounding box to generate a segmentation mask.
    Uses Frangi ridge detection as fallback when edge detection fails.
    
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
        enable_frangi: Whether to enable Frangi fallback
        frangi_sigmas: List of scales for Frangi filter
        frangi_beta1: Frangi parameter for line-like structures
        frangi_beta2: Frangi parameter to suppress blob-like responses
        frangi_threshold_percentile: Percentile threshold for Frangi response
        frangi_threshold_absolute: Absolute threshold (overrides percentile if provided)
        frangi_min_length: Minimum contour length for Frangi components
        frangi_dilation_kernel: Dilation kernel size for Frangi mask
        edge_pixel_threshold: Minimum number of edge pixels to consider edge detection successful
    
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
    
    # Check if edge detection produced sufficient results
    edge_pixel_count = np.sum(filtered_edges > 0)
    use_frangi_fallback = enable_frangi and (edge_pixel_count < edge_pixel_threshold)
    
    if use_frangi_fallback:
        # Use Frangi ridge detection as fallback
        mask_crop = apply_frangi_fallback(
            crop,
            frangi_sigmas=frangi_sigmas,
            frangi_beta1=frangi_beta1,
            frangi_beta2=frangi_beta2,
            frangi_threshold_percentile=frangi_threshold_percentile,
            frangi_threshold_absolute=frangi_threshold_absolute,
            frangi_min_length=frangi_min_length,
            frangi_dilation_kernel=frangi_dilation_kernel
        )
    else:
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


def mask_to_coco_segmentation(mask: np.ndarray, use_rle: bool = True) -> Union[Dict, List]:
    """
    Convert binary mask to COCO segmentation format (RLE or polygon).
    
    Args:
        mask: Binary mask [H, W] with values {0, 255} or {0, 1}
        use_rle: If True, return RLE format; if False, return polygon format
    
    Returns:
        COCO segmentation format (dict with RLE or list of polygons)
    """
    # Ensure mask is binary uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Convert to binary {0, 1}
    mask_binary = (mask > 127).astype(np.uint8)
    
    if not PYCOCOTOOLS_AVAILABLE:
        # Fallback: return empty polygon if pycocotools not available
        return []
    
    if use_rle:
        # Convert to RLE format
        rle = mask_utils.encode(np.asfortranarray(mask_binary))
        # Convert counts from bytes to string for JSON serialization
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    else:
        # Convert to polygon format
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Flatten contour to [x1, y1, x2, y2, ...] format
            if len(contour) >= 3:  # Need at least 3 points for a polygon
                polygon = contour.reshape(-1, 2).flatten().tolist()
                polygons.append(polygon)
        
        return polygons if polygons else []


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
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    edge_detector_type: str = "canny",
    canny_low_threshold: float = 50.0,
    canny_high_threshold: float = 150.0,
    color_difference_threshold: float = 15.0,
    min_edge_length: int = 10,
    dilation_kernel_size: int = 3,
    erosion_kernel_size: int = 2,
    enable_frangi: bool = True,
    frangi_sigmas: List[float] = None,
    frangi_beta1: float = 0.5,
    frangi_beta2: float = 15.0,
    frangi_threshold_percentile: float = 90.0,
    frangi_threshold_absolute: Optional[float] = None,
    frangi_min_length: int = 10,
    frangi_dilation_kernel: int = 2,
    edge_pixel_threshold: int = 10
):
    """
    Main function to process the dataset and generate pseudo masks.
    
    Args:
        data_root: Path to dataset root directory containing images
        annotations_path: Path to COCO format annotations JSON file
        output_dir: Directory to save generated splits and visualizations
        num_images: Number of images to process (None for all)
        train_ratio: Ratio of images for training (default: 0.8)
        valid_ratio: Ratio of images for validation (default: 0.1)
        test_ratio: Ratio of images for testing (default: 0.1)
        edge_detector_type: "canny" or "sobel"
        canny_low_threshold: Lower threshold for Canny edge detector
        canny_high_threshold: Upper threshold for Canny edge detector
        color_difference_threshold: L2 distance threshold in LAB color space
        min_edge_length: Minimum edge length in pixels
        dilation_kernel_size: Dilation kernel size for post-processing
        erosion_kernel_size: Erosion kernel size for post-processing
        enable_frangi: Whether to enable Frangi fallback
        frangi_sigmas: List of scales for Frangi filter
        frangi_beta1: Frangi parameter for line-like structures
        frangi_beta2: Frangi parameter to suppress blob-like responses
        frangi_threshold_percentile: Percentile threshold for Frangi response
        frangi_threshold_absolute: Absolute threshold (overrides percentile if provided)
        frangi_min_length: Minimum contour length for Frangi components
        frangi_dilation_kernel: Dilation kernel size for Frangi mask
        edge_pixel_threshold: Minimum number of edge pixels to consider edge detection successful
    """
    # Validate split ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")
    
    # Create output directories
    output_path = Path(output_dir)
    visualizations_dir = output_path / "visualizations"
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    test_dir = output_path / "test"
    
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Shuffle image IDs for random split
    np.random.shuffle(image_ids)
    
    # Split image IDs into train/valid/test
    total_images = len(image_ids)
    train_end = int(total_images * train_ratio)
    valid_end = train_end + int(total_images * valid_ratio)
    
    train_ids = image_ids[:train_end]
    valid_ids = image_ids[train_end:valid_end]
    test_ids = image_ids[valid_end:]
    
    print(f"Processing {total_images} images...")
    print(f"Split: train={len(train_ids)} ({len(train_ids)/total_images*100:.1f}%), "
          f"valid={len(valid_ids)} ({len(valid_ids)/total_images*100:.1f}%), "
          f"test={len(test_ids)} ({len(test_ids)/total_images*100:.1f}%)")
    
    # Initialize COCO format data structures for each split
    def create_coco_structure():
        return {
            "info": {
                "description": "Pseudo segmentation masks generated from bounding box annotations",
                "version": "1.0"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": coco_data.get("categories", [])
        }
    
    coco_train = create_coco_structure()
    coco_valid = create_coco_structure()
    coco_test = create_coco_structure()
    
    # If no categories exist, create a default one
    if not coco_train["categories"]:
        default_category = [{
            "id": 1,
            "name": "taping_area",
            "supercategory": "drywall"
        }]
        coco_train["categories"] = default_category
        coco_valid["categories"] = default_category
        coco_test["categories"] = default_category
    
    # Get category ID (use first category or default to 1)
    category_id = coco_train["categories"][0]["id"] if coco_train["categories"] else 1
    
    # Track annotation and image IDs per split
    train_annotation_id = 1
    train_image_id = 1
    valid_annotation_id = 1
    valid_image_id = 1
    test_annotation_id = 1
    test_image_id = 1
    
    # Process each image
    all_image_ids = train_ids + valid_ids + test_ids
    for image_id in tqdm(all_image_ids, desc="Processing images"):
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
        
        # Determine which split this image belongs to
        if image_id in train_ids:
            split_dir = train_dir
            coco_output = coco_train
            current_image_id = train_image_id
            current_annotation_id = train_annotation_id
        elif image_id in valid_ids:
            split_dir = valid_dir
            coco_output = coco_valid
            current_image_id = valid_image_id
            current_annotation_id = valid_annotation_id
        else:  # test
            split_dir = test_dir
            coco_output = coco_test
            current_image_id = test_image_id
            current_annotation_id = test_annotation_id
        
        # Copy image to appropriate split directory
        split_img_path = split_dir / img_filename
        # Save image in RGB format (convert back to BGR for OpenCV save)
        cv2.imwrite(str(split_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
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
                erosion_kernel_size=erosion_kernel_size,
                enable_frangi=enable_frangi,
                frangi_sigmas=frangi_sigmas,
                frangi_beta1=frangi_beta1,
                frangi_beta2=frangi_beta2,
                frangi_threshold_percentile=frangi_threshold_percentile,
                frangi_threshold_absolute=frangi_threshold_absolute,
                frangi_min_length=frangi_min_length,
                frangi_dilation_kernel=frangi_dilation_kernel,
                edge_pixel_threshold=edge_pixel_threshold
            )
            
            # Combine masks (union)
            combined_mask = np.maximum(combined_mask, mask)
        
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
        
        # Add to COCO format output
        # Only add if mask has content
        if np.sum(combined_mask > 0) > 0:
            # Add image entry
            coco_output["images"].append({
                "id": current_image_id,
                "width": image.shape[1],
                "height": image.shape[0],
                "file_name": img_filename
            })
            
            # Convert mask to COCO segmentation format (use RLE for efficiency)
            segmentation = mask_to_coco_segmentation(combined_mask, use_rle=True)
            
            # Compute bounding box from mask
            y_coords, x_coords = np.where(combined_mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                x_min = float(np.min(x_coords))
                y_min = float(np.min(y_coords))
                x_max = float(np.max(x_coords))
                y_max = float(np.max(y_coords))
                bbox_coco = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = float(np.sum(combined_mask > 0))
            else:
                bbox_coco = [0.0, 0.0, 0.0, 0.0]
                area = 0.0
            
            # Add annotation entry
            coco_output["annotations"].append({
                "id": current_annotation_id,
                "image_id": current_image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox_coco,
                "iscrowd": 0
            })
            
            # Update counters for the appropriate split
            if image_id in train_ids:
                train_annotation_id += 1
                train_image_id += 1
            elif image_id in valid_ids:
                valid_annotation_id += 1
                valid_image_id += 1
            else:  # test
                test_annotation_id += 1
                test_image_id += 1
    
    # Save COCO format annotations for each split
    train_annotations_path = train_dir / "_annotations.coco.json"
    valid_annotations_path = valid_dir / "_annotations.coco.json"
    test_annotations_path = test_dir / "_annotations.coco.json"
    
    with open(train_annotations_path, 'w') as f:
        json.dump(coco_train, f, indent=2)
    with open(valid_annotations_path, 'w') as f:
        json.dump(coco_valid, f, indent=2)
    with open(test_annotations_path, 'w') as f:
        json.dump(coco_test, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Visualizations saved to: {visualizations_dir}")
    print(f"\nTrain split:")
    print(f"  Images: {train_dir}")
    print(f"  COCO annotations: {train_annotations_path}")
    print(f"  Total images: {len(coco_train['images'])}")
    print(f"  Total annotations: {len(coco_train['annotations'])}")
    print(f"\nValid split:")
    print(f"  Images: {valid_dir}")
    print(f"  COCO annotations: {valid_annotations_path}")
    print(f"  Total images: {len(coco_valid['images'])}")
    print(f"  Total annotations: {len(coco_valid['annotations'])}")
    print(f"\nTest split:")
    print(f"  Images: {test_dir}")
    print(f"  COCO annotations: {test_annotations_path}")
    print(f"  Total images: {len(coco_test['images'])}")
    print(f"  Total annotations: {len(coco_test['annotations'])}")


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
        'train_ratio': 0.8,
        'valid_ratio': 0.1,
        'test_ratio': 0.1,
        'edge_detector_type': 'canny',
        'canny_low_threshold': 50.0,
        'canny_high_threshold': 150.0,
        'color_difference_threshold': 15.0,
        'min_edge_length': 10,
        'dilation_kernel_size': 3,
        'erosion_kernel_size': 2,
        'enable_frangi': True,
        'frangi_sigmas': [1.0, 2.0, 3.0, 4.0, 5.0],
        'frangi_beta1': 0.5,
        'frangi_beta2': 15.0,
        'frangi_threshold_percentile': 90.0,
        'frangi_threshold_absolute': None,
        'frangi_min_length': 10,
        'frangi_dilation_kernel': 2,
        'edge_pixel_threshold': 10
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
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "test_ratio": 0.1,
            "edge_detector_type": "canny",
            "canny_low_threshold": 50.0,
            "canny_high_threshold": 150.0,
            "color_difference_threshold": 15.0,
            "min_edge_length": 10,
            "dilation_kernel_size": 3,
            "erosion_kernel_size": 2,
            "enable_frangi": True,
            "frangi_sigmas": [1.0, 2.0, 3.0, 4.0, 5.0],
            "frangi_beta1": 0.5,
            "frangi_beta2": 15.0,
            "frangi_threshold_percentile": 90.0,
            "frangi_threshold_absolute": None,
            "frangi_min_length": 10,
            "frangi_dilation_kernel": 2,
            "edge_pixel_threshold": 10
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
        train_ratio=config['train_ratio'],
        valid_ratio=config['valid_ratio'],
        test_ratio=config['test_ratio'],
        edge_detector_type=config['edge_detector_type'],
        canny_low_threshold=config['canny_low_threshold'],
        canny_high_threshold=config['canny_high_threshold'],
        color_difference_threshold=config['color_difference_threshold'],
        min_edge_length=config['min_edge_length'],
        dilation_kernel_size=config['dilation_kernel_size'],
        erosion_kernel_size=config['erosion_kernel_size'],
        enable_frangi=config['enable_frangi'],
        frangi_sigmas=config.get('frangi_sigmas'),
        frangi_beta1=config['frangi_beta1'],
        frangi_beta2=config['frangi_beta2'],
        frangi_threshold_percentile=config['frangi_threshold_percentile'],
        frangi_threshold_absolute=config.get('frangi_threshold_absolute'),
        frangi_min_length=config['frangi_min_length'],
        frangi_dilation_kernel=config['frangi_dilation_kernel'],
        edge_pixel_threshold=config['edge_pixel_threshold']
    )


if __name__ == "__main__":
    main()

