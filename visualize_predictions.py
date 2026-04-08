"""
Car Detection Visualization Script
===================================

Visualizes predictions from trained Swin+GLCM car detector model.
Loads a trained model checkpoint and generates visual predictions on input images.

Usage:
    python visualize_predictions.py --image path/to/image.jpg --checkpoint path/to/best_model.pth
    python visualize_predictions.py --image_dir path/to/images/ --checkpoint path/to/best_model.pth --output_dir results/
"""

import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - no popups
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

from swin_detector_model import SwinMask2Former
from car_detector_dataset import CarDetectorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarDetectorVisualizer:
    """Visualizer for car detection predictions"""
    
    def __init__(self, checkpoint_path, device='cuda', confidence_threshold=0.5):
        """
        Initialize the visualizer with a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Device to run inference on ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence score for displaying predictions
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract configuration
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            self.model_config = self.config.get('model_config', {})
        elif 'model_architecture_params' in checkpoint:
            self.model_config = checkpoint['model_architecture_params']
            self.config = {'model_config': self.model_config}
        else:
            raise ValueError("Checkpoint does not contain configuration information")
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = SwinMask2Former(config=self.model_config).to(self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✓ Model weights loaded successfully")
        else:
            raise ValueError("Checkpoint does not contain model_state_dict")
        
        self.model.eval()
        
        # Extract metadata
        self.num_classes = self.model_config.get('num_classes', 1)
        self.img_size = self.model_config.get('img_size', 512)
        self.use_glcm = self.model_config.get('use_glcm', False)
        
        # Create validation transform (matching training configuration exactly)
        self.transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Log model info
        if 'best_val_map' in checkpoint:
            logger.info(f"Model Best Validation AP: {checkpoint['best_val_map']:.4f}")
        if 'epoch' in checkpoint:
            logger.info(f"Model trained for {checkpoint['epoch'] + 1} epochs")
        
        logger.info(f"Model configuration:")
        logger.info(f"  - Image size: {self.img_size}×{self.img_size}")
        logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Use GLCM: {self.use_glcm}")
        logger.info(f"  - Confidence threshold: {self.confidence_threshold}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input (matching training configuration exactly).
        
        Args:
            image_path: Path to input image
            
        Returns:
            tuple: (preprocessed_tensor, resized_image, texture_map)
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB (matching training dataset)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply validation transform (Resize + Normalize + ToTensor)
        # This exactly matches the val_test_transform in car_detector_dataset.py
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension (1, C, H, W)
        
        # Create resized image for visualization (512×512 in RGB)
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Create dummy texture map (8 channels, 32×32 grid)
        # In production, you would compute actual GLCM features if use_glcm=True
        texture_map = torch.zeros(1, 8, 32, 32)
        
        # Return resized image in BGR format for OpenCV compatibility
        return image_tensor, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), texture_map
    
    def predict(self, image_tensor, texture_map):
        """
        Run inference on preprocessed image.
        
        Args:
            image_tensor: Preprocessed image tensor (1, C, H, W)
            texture_map: Texture map tensor (1, 8, 32, 32)
            
        Returns:
            dict: Model outputs with pred_logits and pred_masks
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            texture_map = texture_map.to(self.device)
            
            outputs = self.model(pixel_values=image_tensor, texture_maps=texture_map)
        
        return outputs
    
    def postprocess_predictions(self, outputs, original_shape):
        """
        Post-process model outputs to get final predictions.
        
        Args:
            outputs: Model outputs dictionary
            original_shape: Original image shape (H, W)
            
        Returns:
            tuple: (masks_list, scores_list, class_ids_list)
        """
        pred_logits = outputs['pred_logits'][0].cpu()  # (Q, num_classes+1)
        pred_masks = outputs['pred_masks'][0].cpu()    # (Q, H, W)
        
        # Get class probabilities
        probs = pred_logits.softmax(-1)  # (Q, num_classes+1)
        eos_idx = self.num_classes       # EOS/background class index
        
        # Compute objectness score (1 - P(EOS))
        objectness = 1.0 - probs[:, eos_idx]
        
        # Get predicted class (excluding EOS)
        cls_probs, cls_ids = probs[:, :eos_idx].max(dim=-1)
        
        # Filter by confidence threshold
        keep = objectness > self.confidence_threshold
        
        if not keep.any():
            return [], [], []
        
        # Get filtered predictions
        filtered_masks = pred_masks[keep]
        filtered_scores = objectness[keep]
        filtered_classes = cls_ids[keep]
        
        # Upsample masks to original image size
        H, W = original_shape
        upsampled_masks = F.interpolate(
            filtered_masks.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Apply sigmoid and threshold
        binary_masks = (upsampled_masks.sigmoid() > 0.5).numpy()
        scores = filtered_scores.numpy()
        class_ids = filtered_classes.numpy()
        
        return binary_masks, scores, class_ids
    
    def visualize_prediction(self, image, masks, scores, class_ids):
        """
        Visualize predictions on image using simple contour drawing (matching training style).
        
        Args:
            image: Original image (BGR format)
            masks: List of binary masks
            scores: Confidence scores
            class_ids: Class IDs
            
        Returns:
            numpy.ndarray: Visualized image
        """
        # Convert BGR to RGB for consistency
        vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        
        if len(masks) == 0:
            # No detections - return original image
            return vis_image
        
        # Draw predicted masks with red contours (matching training validation style)
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            mask_np = mask.astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Draw contours in red (matching validation visualization)
            cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 1)
            
            # Optionally add confidence score at centroid
            # M = cv2.moments(contours[0])
            # if M["m00"] > 0:
            #     cX = int(M["m10"] / M["m00"])
            #     cY = int(M["m01"] / M["m00"])
            #     # Small text with confidence score
            #     cv2.putText(vis_image, f"{score:.2f}", (cX - 15, cY),
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return vis_image
    
    def process_image(self, image_path, output_path=None, show_plot=True):
        """
        Process single image and generate visualization.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            show_plot: Whether to display result with matplotlib
            
        Returns:
            tuple: (visualized_image, num_detections)
        """
        logger.info(f"Processing: {image_path}")
        
        # Preprocess (returns 512×512 resized image)
        image_tensor, resized_image, texture_map = self.preprocess_image(image_path)
        
        # Predict
        outputs = self.predict(image_tensor, texture_map)
        
        # Post-process (use resized image dimensions: 512×512)
        H, W = resized_image.shape[:2]
        masks, scores, class_ids = self.postprocess_predictions(outputs, (H, W))
        
        logger.info(f"  Detected {len(masks)} car(s)")
        
        # Visualize on resized image
        vis_image = self.visualize_prediction(
            resized_image, masks, scores, class_ids
        )
        
        # Save if output path provided
        if output_path:
            # Create parent directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert RGB back to BGR for saving with cv2
            cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            logger.info(f"  Saved to: {output_path}")
        
        # Display if requested
        if show_plot:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.title("Input Image (512×512)")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(vis_image)
            plt.title(f"Predictions ({len(masks)} car(s) detected)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return vis_image, len(masks)
    
    def process_directory(self, input_dir, output_dir, extensions=('.jpg', '.jpeg', '.png')):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            extensions: Tuple of valid image extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for img_file in tqdm(image_files, desc="Processing images"):
            try:
                output_file = output_path / f"{img_file.stem}_prediction{img_file.suffix}"
                vis_image, num_detections = self.process_image(
                    img_file, 
                    output_file, 
                    show_plot=False
                )
                
                results.append({
                    'image': img_file.name,
                    'num_detections': num_detections,
                    'output': output_file.name
                })
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
        
        # Save summary
        summary_file = output_path / "detection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_images': len(image_files),
                'processed': len(results),
                'results': results
            }, f, indent=2)
        
        logger.info(f"\n✓ Processing complete!")
        logger.info(f"  Total images: {len(image_files)}")
        logger.info(f"  Successfully processed: {len(results)}")
        logger.info(f"  Results saved to: {output_dir}")
        logger.info(f"  Summary saved to: {summary_file}")


def main():
    """Main entry point for visualization script"""
    parser = argparse.ArgumentParser(
        description='Visualize predictions from trained car detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python visualize_predictions.py --image test.jpg --checkpoint checkpoints/best_model.pth
  
  # Process directory of images
  python visualize_predictions.py --image_dir test_images/ --checkpoint checkpoints/best_model.pth --output_dir results/
  
  # Adjust confidence threshold
  python visualize_predictions.py --image test.jpg --checkpoint checkpoints/best_model.pth --confidence 0.7
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, default=r'../../car_detection_output_20251030_103528\checkpoints\latest_checkpoint.pth',
                       help='Path to model checkpoint (.pth file)')
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                            help='Path to single input image')
    input_group.add_argument('--image_dir', type=str,
                            help='Directory containing input images')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output path for single image (default: <image>_prediction.jpg)')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='Output directory for batch processing (default: visualization_results)')
    
    # Visualization options
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display results with matplotlib (useful for batch processing)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on (default: cuda)')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    try:
        # Initialize visualizer
        visualizer = CarDetectorVisualizer(
            checkpoint_path=args.checkpoint,
            device=args.device,
            confidence_threshold=args.confidence
        )
        
        # Process based on input type
        if args.image:
            # Single image processing
            output_path = args.output
            if output_path is None:
                # Generate default output name
                img_path = Path(args.image)
                output_path = f"{args.output_dir}/{img_path.stem}_prediction{img_path.suffix}"
            
            visualizer.process_image(
                args.image,
                output_path=output_path,
                show_plot=not args.no_display
            )
            
        else:
            # Directory processing
            if not os.path.exists(args.image_dir):
                logger.error(f"Input directory not found: {args.image_dir}")
                return
            
            visualizer.process_directory(
                args.image_dir,
                args.output_dir
            )
        
        logger.info("\n✓ Visualization complete!")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
