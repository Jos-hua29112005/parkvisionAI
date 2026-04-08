"""
Car Detector Dataset for Swin+GLCM Car Detection
===============================================

This module provides dataset functionality for training a car detector using:
1. Full-frame images for detection
2. Car bounding boxes from label files
3. GLCM texture feature extraction
4. Support for polygon-to-bbox conversion
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict
import logging

from glcm_utils import compute_local_glcm_map, compute_local_glcm_map_cached

logger = logging.getLogger(__name__)

# NEW: Function to convert polygons to a stack of binary masks
def polygons_to_masks(polygons: List[np.ndarray], height: int, width: int) -> torch.Tensor:
    """Converts a list of polygons into a tensor of binary instance masks."""
    masks = []
    for poly in polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        # De-normalize polygon points from [0, 1] to pixel coordinates
        poly_pixels = (poly * np.array([width, height])).astype(np.int32)
        cv2.fillPoly(mask, [poly_pixels], 1)
        masks.append(torch.from_numpy(mask).bool())
        
    if not masks:
        return torch.empty((0, height, width), dtype=torch.bool)
        
    return torch.stack(masks, dim=0)

# Top-level collate function to be picklable on Windows (spawn)
def detection_collate_fn(batch):
    """Batch images, texture maps, and structured targets."""
    images, texture_maps, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    texture_maps = torch.stack(texture_maps, dim=0)
    return images, texture_maps, list(targets)


class CarDetectorDataset(Dataset):
    """
    Dataset for car detection using full-frame images and bounding box labels
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform=None,
        img_size: int = 224,
        use_glcm: bool = True,
        texture_config: Dict | None = None,
        cache_glcm: bool = True,
        glcm_cache_dir: str | None = None,
    ):
        """
        Initialize the car detector dataset with GLOBAL GLCM features.
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files with car bounding boxes
            transform: Image transformations (applied to both image and masks)
            img_size: Target image size for augmentations and GLCM extraction
            use_glcm: Whether to use GLCM features (default: True)
            cache_glcm: Whether to cache GLCM maps on disk (default: True)
            glcm_cache_dir: Directory for GLCM cache (default: <images_dir>/.glcm_cache)
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_size = img_size
        self.use_glcm = use_glcm
        self.cache_glcm = cache_glcm
        
        # Setup GLCM cache directory
        if glcm_cache_dir is None:
            self.glcm_cache_dir = os.path.join(os.path.dirname(images_dir), '.glcm_cache')
        else:
            self.glcm_cache_dir = glcm_cache_dir
        
        default_texture = {
            'grid_size': 32,
            'distances': [1],
            'angles': [0.0, np.pi / 2.0],
            'levels': 32,
            'properties': ['contrast', 'homogeneity', 'energy', 'correlation']
        }
        self.texture_config = texture_config or default_texture
        self.texture_grid = int(self.texture_config.get('grid_size', 32))
        self.texture_distances = tuple(self.texture_config.get('distances', [1]))
        self.texture_angles = tuple(self.texture_config.get('angles', [0.0, np.pi / 2.0]))
        self.texture_levels = int(self.texture_config.get('levels', 32))
        self.texture_properties = tuple(self.texture_config.get('properties', ['contrast', 'homogeneity', 'energy', 'correlation']))
        
        self.samples = []
        self._load_samples()

        logger.info(f"Car detector dataset initialized with {len(self.samples)} samples")
        if self.use_glcm:
            feature_count = len(self.texture_distances) * len(self.texture_angles) * len(self.texture_properties)
            logger.info(
                "Using local GLCM texture maps: %d channels over a %dx%d grid",
                feature_count,
                self.texture_grid,
                self.texture_grid,
            )
    
    def _load_samples(self):
        """Load image and label file pairs"""
        image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))
        
        for img_path in image_files:
            base = os.path.basename(img_path).replace('.jpg', '.txt')
            label_path = os.path.join(self.labels_dir, base)
            
            if os.path.exists(label_path):
                self.samples.append({
                    'image_path': img_path,
                    'label_path': label_path
                })
    
    def _compute_texture_map(self, rgb_tensor: torch.Tensor, image_path: str) -> torch.Tensor:
        if not self.use_glcm:
            return torch.zeros((0, 0, 0), dtype=torch.float32)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=rgb_tensor.dtype).view(3, 1, 1)
        restored = (rgb_tensor * std + mean).clamp(0.0, 1.0)
        rgb_np = (restored.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        if self.cache_glcm:
            local_map = compute_local_glcm_map_cached(
                rgb_np,
                image_path=image_path,
                cache_dir=self.glcm_cache_dir,
                grid_cells=self.texture_grid,
                distances=self.texture_distances,
                angles=self.texture_angles,
                levels=self.texture_levels,
                properties=self.texture_properties,
            )
        else:
            local_map = compute_local_glcm_map(
                rgb_np,
                grid_cells=self.texture_grid,
                distances=self.texture_distances,
                angles=self.texture_angles,
                levels=self.texture_levels,
                properties=self.texture_properties,
            )

        return torch.from_numpy(local_map).float()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample from the dataset - UPDATED FOR MASKS
        
        Returns:
            Tuple of (image_tensor, glcm_features_tensor, target_dict)
            where target_dict now contains 'masks' instead of 'boxes'
        """
        sample_info = self.samples[idx]
        
        try:
            # Load image
            image = cv2.imread(sample_info['image_path'])
            if image is None:
                raise ValueError(f"Could not load image: {sample_info['image_path']}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]

            # 1. Parse polygons and labels from the file
            polygons = []
            labels = []
            with open(sample_info['label_path'], 'r') as f:
                for line in f:
                    data = list(map(float, line.strip().split()))
                    if len(data) < 7: continue
                    polygons.append(np.array(data[1:]).reshape(-1, 2))
                    labels.append(int(data[0]))

            # 3. Convert polygons into a stack of binary masks (as numpy arrays for augmentation)
            masks_np = polygons_to_masks(polygons, img_height, img_width).numpy().astype(np.uint8)

            # 4. Apply augmentations to both image and masks
            if self.transform:
                try:
                    transformed = self.transform(image=image, masks=list(masks_np))
                    image_tensor = transformed['image']

                    if transformed['masks']:
                        masks = torch.from_numpy(np.stack(transformed['masks'])).bool()
                    else:
                        _, h, w = image_tensor.shape
                        masks = torch.empty((0, h, w), dtype=torch.bool)
                except Exception as e:
                    logger.warning(f"Transform failed for sample {idx}: {e}, keeping original")
                    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                    masks = torch.from_numpy(masks_np).bool() if len(masks_np) else torch.empty((0, img_height, img_width), dtype=torch.bool)
            else:
                image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                masks = torch.from_numpy(masks_np).bool() if len(masks_np) else torch.empty((0, img_height, img_width), dtype=torch.bool)

            texture_map = self._compute_texture_map(image_tensor, sample_info['image_path'])

            target = {
                'labels': torch.tensor(labels, dtype=torch.int64),
                'masks': masks
            }
            
            return image_tensor, texture_map, target
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({sample_info.get('image_path', 'N/A')}): {e}")
            # Return dummy sample with masks and GLOBAL GLCM
            dummy_image = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            if self.use_glcm:
                feature_count = len(self.texture_distances) * len(self.texture_angles) * len(self.texture_properties)
                dummy_map = torch.zeros((feature_count, self.texture_grid, self.texture_grid), dtype=torch.float32)
            else:
                dummy_map = torch.zeros((0, 0, 0), dtype=torch.float32)
            dummy_target = {
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, self.img_size, self.img_size), dtype=torch.bool)
            }
            return dummy_image, dummy_map, dummy_target
    
def create_detector_dataloaders(
    base_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 512,
    use_glcm: bool = True,
    texture_config: Dict | None = None,
):
    """
    Create dataloaders for car detection training, validation, and testing with GLOBAL GLCM.
    
    Args:
        base_dir: Base directory containing train/val/test splits
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        img_size: Target image size for resizing and tensor shapes (default: 512 for better detail)
        use_glcm: Whether to use GLCM features (default: True)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    import os
    
    # NEW: More Aggressive Augmentations for Better Generalization
    train_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.7),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Keep validation simple - no augmentation except normalization
    val_test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    splits = ['train', 'val', 'test']
    datasets = {}
    
    for split in splits:
        images_dir = os.path.join(base_dir, split, 'images')
        labels_dir = os.path.join(base_dir, split, 'labels')
        
        # Check if directories exist
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            continue
            
        if not os.path.exists(labels_dir):
            logger.warning(f"Labels directory not found: {labels_dir}")
            continue
            
        transform = train_transform if split == 'train' else val_test_transform
        
        try:
            dataset = CarDetectorDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                transform=transform,
                img_size=img_size,
                use_glcm=use_glcm,
                texture_config=texture_config,
            )
            datasets[split] = dataset
            logger.info(f"Created {split} dataset with {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to create {split} dataset: {e}")
    
    # Create dataloaders
    loaders = {}
    for split in splits:
        if split in datasets:
            shuffle = (split == 'train')
            loaders[split] = DataLoader(
                datasets[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=detection_collate_fn,
                pin_memory=torch.cuda.is_available()
            )
        else:
            loaders[split] = None
    
    return loaders.get('train'), loaders.get('val'), loaders.get('test')


if __name__ == "__main__":
    # Example usage
    dataset = CarDetectorDataset(
        images_dir="../../data/Combined",
        labels_dir="../../data/labels_comb"
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Test loading a sample
    if len(dataset) > 0:
        image, texture_map, target = dataset[0]
        print(f"Sample 0: Image shape: {image.shape}")
        print(f"Texture map shape: {texture_map.shape}")
        print(f"Labels shape: {target['labels'].shape}")
    
    # Test dataloader creation
    try:
        train_loader, val_loader, test_loader = create_detector_dataloaders(
            base_dir="../../split_data/",
            batch_size=4,
            num_workers=0  # Set to 0 for debugging
        )
        print("✅ Dataloaders created successfully")
        
        if train_loader:
            # Test loading a batch
            for batch_images, batch_texture, batch_targets in train_loader:
                print(f"Batch size: {len(batch_images)}")
                print(f"Image batch shape: {batch_images.shape}")
                print(f"Texture map batch shape: {batch_texture.shape}")
                print(f"Number of targets: {len(batch_targets)}")
                break
                
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
