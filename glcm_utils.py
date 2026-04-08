# glcm_utils.py
import numpy as np
import cv2
import os
import hashlib
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops


def calculate_global_glcm_features(image_path, img_size, distances, angles, levels=256):
    """Backward-compatible helper to compute a single global GLCM vector."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Could not read image at {image_path}")
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        num_props = 6
        num_features = len(distances) * len(angles) * num_props
        print(f"Error processing image {image_path}: {exc}")
        return np.zeros(num_features, dtype=np.float32)

    if np.max(gray_image) == np.min(gray_image):
        num_props = 6
        num_features = len(distances) * len(angles) * num_props
        return np.zeros(num_features, dtype=np.float32)

    glcm = graycomatrix(
        gray_image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    props = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        props.append(graycoprops(glcm, prop).flatten())

    return np.concatenate(props).astype(np.float32)


def compute_local_glcm_map(
    image: np.ndarray,
    grid_cells: int = 32,
    *,
    distances=(1,),
    angles=(0.0, np.pi / 2.0),
    levels: int = 32,
    properties=("contrast", "homogeneity", "energy", "correlation"),
) -> np.ndarray:
    """Compute a lightweight grid of local GLCM descriptors.

    Args:
        image: Input image as RGB, BGR, or grayscale numpy array (H, W[, C]).
        grid_cells: Number of cells along each spatial dimension (e.g. 32 => 32×32 grid).
        distances: Iterable of pixel pair distances.
        angles: Iterable of angles (in radians) for GLCM computation.
        levels: Gray-level quantisation used before constructing the GLCM.
        properties: Iterable of GLCM properties to aggregate per cell.

    Returns:
        np.ndarray: Array with shape (num_features, grid_cells, grid_cells) where
            num_features = len(distances) * len(angles) * len(properties).
    """

    if grid_cells < 1:
        raise ValueError("grid_cells must be >= 1")

    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray_image = image
    else:
        raise ValueError("Image must be H×W or H×W×C array")

    gray_image = np.asarray(gray_image, dtype=np.float32)

    if gray_image.max() <= 1.0:
        gray_image = gray_image * 255.0

    gray_image = gray_image.astype(np.uint8)

    height, width = gray_image.shape
    cell_h = max(height // grid_cells, 1)
    cell_w = max(width // grid_cells, 1)

    feature_count = len(distances) * len(angles) * len(properties)
    texture_grid = np.zeros((feature_count, grid_cells, grid_cells), dtype=np.float32)

    quantisation_scale = (levels - 1) / 255.0 if levels > 1 else 1.0

    for row in range(grid_cells):
        top = row * cell_h
        bottom = height if row == grid_cells - 1 else (row + 1) * cell_h

        for col in range(grid_cells):
            left = col * cell_w
            right = width if col == grid_cells - 1 else (col + 1) * cell_w

            patch = gray_image[top:bottom, left:right]

            if patch.size == 0 or patch.max() == patch.min():
                # Leave the feature vector as zeros when the patch is uniform.
                continue

            quantised_patch = np.clip(patch.astype(np.float32) * quantisation_scale, 0, levels - 1).astype(np.uint8)

            glcm = graycomatrix(
                quantised_patch,
                distances=distances,
                angles=angles,
                levels=levels,
                symmetric=True,
                normed=True,
            )

            feature_idx = 0
            for prop in properties:
                prop_values = graycoprops(glcm, prop).astype(np.float32)
                flat_values = prop_values.flatten()
                length = flat_values.shape[0]
                texture_grid[feature_idx:feature_idx + length, row, col] = flat_values
                feature_idx += length

    return texture_grid


def compute_local_glcm_map_cached(
    image: np.ndarray,
    image_path: str,
    cache_dir: str,
    grid_cells: int = 32,
    *,
    distances=(1,),
    angles=(0.0, np.pi / 2.0),
    levels: int = 32,
    properties=("contrast", "homogeneity", "energy", "correlation"),
) -> np.ndarray:
    """Compute local GLCM map with disk caching.
    
    Args:
        image: Input image as RGB, BGR, or grayscale numpy array (H, W[, C]).
        image_path: Path to the source image file (used for cache key).
        cache_dir: Directory to store cached GLCM maps.
        grid_cells: Number of cells along each spatial dimension.
        distances: Iterable of pixel pair distances.
        angles: Iterable of angles (in radians) for GLCM computation.
        levels: Gray-level quantisation used before constructing the GLCM.
        properties: Iterable of GLCM properties to aggregate per cell.
    
    Returns:
        np.ndarray: Array with shape (num_features, grid_cells, grid_cells).
    """
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key from image path and parameters
    param_str = f"{grid_cells}_{distances}_{angles}_{levels}_{properties}"
    cache_key = hashlib.md5(f"{image_path}_{param_str}".encode()).hexdigest()
    cache_file = cache_path / f"{cache_key}.npy"
    
    # Try to load from cache
    if cache_file.exists():
        try:
            return np.load(cache_file)
        except Exception:
            # If cache is corrupted, recompute
            pass
    
    # Compute GLCM map
    texture_map = compute_local_glcm_map(
        image,
        grid_cells=grid_cells,
        distances=distances,
        angles=angles,
        levels=levels,
        properties=properties,
    )
    
    # Save to cache
    try:
        np.save(cache_file, texture_map)
    except Exception as e:
        print(f"Warning: Failed to cache GLCM map: {e}")
    
    return texture_map
