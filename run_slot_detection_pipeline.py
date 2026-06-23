"""
Complete Parking Slot Detection Pipeline Runner
================================================

This script integrates all components to detect empty parking slots:
1. Perspective correction (perspective_correction.py)
2. Car detection with Swin+GLCM (visualize_predictions.py)
3. Polygon transformation to top-down view
4. Slot finding (simplified_slot_finder.py)
5. Visualization of results

Usage:
    python run_slot_detection_pipeline.py --image input.jpg --car_checkpoint model.pth --yolo_model best.pt
"""

import os
import argparse
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, GeometryCollection
from shapely.validation import make_valid

# Import pipeline components
from perspective_correction import load_yolo_model, process_parking_lot_image
from visualize_predictions import CarDetectorVisualizer
from simplified_slot_finder import create_slot_finder
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable matplotlib interactive mode for web deployment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class SlotDetectionPipeline:
    """Complete pipeline for parking slot detection"""
    
    def __init__(self, car_detector_checkpoint, yolo_model_path, 
                 slot_config=None, device='cuda', use_yolo_detection=False, 
                 yolo_car_model_path=None, use_top_down=True):
        """
        Initialize the pipeline with all necessary models.
        
        Args:
            car_detector_checkpoint: Path to Swin+GLCM car detector checkpoint
            yolo_model_path: Path to YOLO parking lot segmentation model
            slot_config: Configuration for slot finder (dict with keys: 
                        standard_slot_width, standard_slot_height, 
                        min_slot_area, max_slot_area)
            device: Device for inference ('cuda' or 'cpu')
            use_yolo_detection: If True, use YOLO for car detection; if False, use Swin+GLCM (default: False)
            yolo_car_model_path: Path to YOLO car segmentation model (required if use_yolo_detection=True)
            use_top_down: If True, apply perspective transformation; if False, use original image (default: True)
        """
        self.device = device
        self.use_yolo_detection = use_yolo_detection
        self.use_top_down = use_top_down
        
        # Load YOLO model for parking lot detection
        logger.info("Loading YOLO model for parking lot detection...")
        load_yolo_model(yolo_model_path)
        
        # Load car detector based on selection
        if use_yolo_detection:
            if yolo_car_model_path is None:
                raise ValueError("yolo_car_model_path must be provided when use_yolo_detection=True")
            logger.info("Loading YOLOv8 segmentation model for car detection...")
            self.car_detector = YOLO(yolo_car_model_path)
            self.car_detector.to(self.device)
            self.confidence_threshold = 0.25
        else:
            logger.info("Loading Swin+GLCM car detector...")
            self.car_detector = CarDetectorVisualizer(
                checkpoint_path=car_detector_checkpoint,
                device=device,
                confidence_threshold=0.5
            )
            self.confidence_threshold = 0.5
        
        # Initialize slot finder with configuration
        logger.info("Initializing slot finder...")
        if use_top_down:
            # Use static configuration for top-down view
            if slot_config is None:
                slot_config = {
                    'standard_slot_width': 162,
                    'standard_slot_height': 57,
                    'min_slot_area': 6687,
                    'max_slot_area': 28657
                }
            self.slot_finder = create_slot_finder(slot_config)
        else:
            # Use dynamic sizing for original image (no top-down)
            # Merge slot_config with dynamic sizing flag
            dynamic_config = {'use_dynamic_sizing': True}
            if slot_config:
                dynamic_config.update(slot_config)
            self.slot_finder = create_slot_finder(dynamic_config)
        
        logger.info("✓ Pipeline initialized successfully")
    
    def get_top_down_view(self, image_path, epsilon_factor=0.02, 
                          visualize=False, output_dir='pipeline_outputs'):
        """
        Step 1: Get perspective-corrected top-down view.
        
        Args:
            image_path: Path to input image
            epsilon_factor: Corner detection sensitivity
            visualize: Whether to save visualizations
            output_dir: Directory for outputs
            
        Returns:
            tuple: (top_down_view, transform_matrix, corners, parking_lot_polygon)
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Perspective Correction")
        logger.info("="*60)
        
        top_down_view, transform_matrix, corners = process_parking_lot_image(
            image_path=image_path,
            epsilon_factor=epsilon_factor,
            visualize=visualize,
            output_dir=output_dir
        )
        
        if top_down_view is None:
            raise ValueError("Failed to create top-down view. Check YOLO detection.")
        
        # Create parking lot boundary polygon for filtering cars
        parking_lot_polygon = Polygon(corners)
        logger.info(f"✓ Top-down view created: {top_down_view.shape}")
        logger.info(f"✓ Parking lot boundary polygon created with {len(corners)} corners")
        
        return top_down_view, transform_matrix, corners, parking_lot_polygon
    
    def detect_cars_in_original(self, image_path, parking_lot_polygon=None):
        """
        Step 2: Detect cars in original image and get binary masks.
        
        Args:
            image_path: Path to input image
            parking_lot_polygon: Shapely Polygon defining parking lot boundaries (optional)
            
        Returns:
            tuple: (masks, scores, original_image)
        """
        logger.info("\n" + "="*60)
        if self.use_yolo_detection:
            logger.info("STEP 2: Car Detection in Original Image (YOLOv8-Seg)")
        else:
            logger.info("STEP 2: Car Detection in Original Image (Swin+GLCM)")
        logger.info("="*60)
        
        # Load original image
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        original_shape = original_image.shape[:2]
        logger.info(f"Original image shape: {original_shape}")
        
        if self.use_yolo_detection:
            # YOLO detection
            results = self.car_detector.predict(
                source=original_image,
                classes=[0],  # Class 2 is 'car' in COCO dataset
                conf=0.75,
                imgsz=640,
                iou=0.7,
                verbose=True,
                max_det=75
            )
            
            result = results[0]
            masks = []
            scores = []
            
            if result.masks is not None:
                conf_scores = result.boxes.conf.cpu().numpy()
                
                for i, mask_tensor in enumerate(result.masks.data):
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(
                        mask_np,
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                    masks.append(mask_resized)
                    scores.append(conf_scores[i])
                
                logger.info(f"✓ Detected {len(masks)} cars with YOLO")
            else:
                logger.warning("No cars detected by YOLO!")
        else:
            # Swin+GLCM detection
            # Preprocess image (resizes to model input size)
            image_tensor, resized_image, texture_map = self.car_detector.preprocess_image(image_path)
            
            # Run inference
            outputs = self.car_detector.predict(image_tensor, texture_map)
            
            # Post-process to get masks at original resolution
            masks, scores, class_ids = self.car_detector.postprocess_predictions(
                outputs, 
                original_shape  # Use original image dimensions
            )
            
            logger.info(f"✓ Detected {len(masks)} cars with Swin+GLCM (confidence > {self.confidence_threshold})")
        
        # Filter cars that are inside the parking lot boundary
        if parking_lot_polygon is not None:
            filtered_masks = []
            filtered_scores = []
            
            for idx, mask in enumerate(masks):
                # Get centroid of the car mask
                mask_uint8 = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Calculate moments to find centroid
                    M = cv2.moments(contours[0])
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Check if centroid is inside parking lot polygon
                        from shapely.geometry import Point
                        car_point = Point(cx, cy)
                        
                        if parking_lot_polygon.contains(car_point):
                            filtered_masks.append(mask)
                            filtered_scores.append(scores[idx])
                        else:
                            logger.debug(f"Car {idx} centroid ({cx}, {cy}) is outside parking lot - filtered out")
            
            cars_before = len(masks)
            masks = filtered_masks
            scores = filtered_scores
            cars_after = len(masks)
            
            logger.info(f"✓ Filtered cars: {cars_before} detected → {cars_after} inside parking lot ({cars_before - cars_after} outside)")
        
        return masks, scores, original_image
    
    def masks_to_polygons(self, masks):
        """
        Step 3: Convert binary masks to polygon contours.
        
        Args:
            masks: List of binary masks (numpy arrays)
            
        Returns:
            list: List of polygon contours (numpy arrays of points)
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Converting Masks to Polygons")
        logger.info("="*60)
        
        car_polygons = []
        
        for idx, mask in enumerate(masks):
            mask_uint8 = mask.astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                logger.warning(f"No contours found for mask {idx}")
                continue
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) < 3:
                logger.warning(f"Contour {idx} has fewer than 3 points")
                continue
            
            car_polygons.append(largest_contour)
        
        logger.info(f"✓ Converted {len(car_polygons)} masks to polygons")
        return car_polygons
    
    def transform_polygons_to_top_down(self, car_polygons, transform_matrix, top_down_shape):
        """
        Step 4: Transform car polygons to top-down view coordinates.
        
        Args:
            car_polygons: List of polygon contours in original image coordinates
            transform_matrix: Perspective transformation matrix from step 1
            top_down_shape: Shape of top-down view (height, width) for boundary checking
            
        Returns:
            list: List of Shapely Polygon objects in top-down coordinates
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Transforming Polygons to Top-Down View")
        logger.info("="*60)
        
        warped_car_polygons = []
        H, W = top_down_shape[:2]
        top_down_boundary = Polygon([(0, 0), (W, 0), (W, H), (0, H)])
        
        for idx, polygon_pts in enumerate(car_polygons):
            try:
                # Reshape for cv2.perspectiveTransform
                pts = polygon_pts.astype(np.float32).reshape(-1, 1, 2)
                
                # Apply perspective transformation
                warped_pts = cv2.perspectiveTransform(pts, transform_matrix)
                
                # Convert to Shapely polygon
                coords = warped_pts.reshape(-1, 2)
                
                if len(coords) < 3:
                    logger.warning(f"Warped polygon {idx} has fewer than 3 points")
                    continue
                
                shapely_poly = Polygon(coords)
                
                # Validate and fix invalid polygons
                if not shapely_poly.is_valid:
                    logger.warning(f"Polygon {idx} is invalid, attempting to fix...")
                    shapely_poly = make_valid(shapely_poly)
                    
                    # make_valid can return GeometryCollection, extract Polygons
                    if isinstance(shapely_poly, GeometryCollection):
                        # Extract the largest polygon from the collection
                        polygons = [geom for geom in shapely_poly.geoms if isinstance(geom, Polygon)]
                        if polygons:
                            shapely_poly = max(polygons, key=lambda p: p.area)
                            logger.info(f"Polygon {idx} extracted from GeometryCollection")
                        else:
                            logger.warning(f"Polygon {idx} GeometryCollection contains no valid polygons")
                            continue
                
                # Check if polygon is within top-down view boundaries
                if isinstance(shapely_poly, Polygon) and shapely_poly.is_valid and not shapely_poly.is_empty:
                    # Check if the polygon intersects with top-down view
                    if shapely_poly.intersects(top_down_boundary):
                        # Keep only the part within bounds
                        shapely_poly = shapely_poly.intersection(top_down_boundary)
                        
                        # After intersection, we might get a GeometryCollection again
                        if isinstance(shapely_poly, GeometryCollection):
                            polygons = [geom for geom in shapely_poly.geoms if isinstance(geom, Polygon)]
                            if polygons:
                                shapely_poly = max(polygons, key=lambda p: p.area)
                            else:
                                logger.warning(f"Polygon {idx} has no valid area after boundary clipping")
                                continue
                        
                        if isinstance(shapely_poly, Polygon) and not shapely_poly.is_empty:
                            warped_car_polygons.append(shapely_poly)
                        else:
                            logger.warning(f"Polygon {idx} is outside top-down view bounds")
                    else:
                        logger.debug(f"Polygon {idx} is completely outside top-down view - filtered out")
                else:
                    logger.warning(f"Polygon {idx} could not be validated")
                    
            except Exception as e:
                logger.error(f"Error transforming polygon {idx}: {e}")
                continue
        
        logger.info(f"✓ Transformed {len(warped_car_polygons)} polygons to top-down view")
        return warped_car_polygons
    
    def find_empty_slots(self, top_down_view, warped_car_polygons, debug_dir=None, valid_column_centers=None):
        """
        Step 5: Find parking slots (both occupied and empty).
        
        Args:
            top_down_view: Top-down view image
            warped_car_polygons: List of Shapely Polygon objects for cars
            debug_dir: Optional directory for debug output
            
        Returns:
            list: List of slot dictionaries with occupancy classification
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Finding Parking Slots")
        logger.info("="*60)
        
        # Returns ALL slots (both occupied and empty) with is_occupied flag
        slots = self.slot_finder.find_slots(top_down_view, warped_car_polygons, debug_dir=debug_dir, valid_column_centers=valid_column_centers)
        
        # Count occupied vs empty
        occupied_count = sum(1 for s in slots if s['is_occupied'])
        empty_count = sum(1 for s in slots if not s['is_occupied'])
        
        logger.info(f"✓ Found {len(slots)} total slots: {occupied_count} occupied, {empty_count} empty")
        return slots
    
    def visualize_results(self, top_down_view, warped_car_polygons, 
                         slots, output_path=None, show_plot=True):
        """
        Step 6: Visualize detection results.
        
        Args:
            top_down_view: Top-down view image
            warped_car_polygons: List of car polygons (for outline reference)
            slots: List of slot dictionaries with occupancy classification
            output_path: Path to save visualization
            show_plot: Whether to display plot
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 6: Visualizing Results")
        logger.info("="*60)
        
        # Create visualization
        vis_image = top_down_view.copy()

        # Draw cars as green outlines only (not filled) to keep slots visible
        for car_poly in warped_car_polygons:
            coords = np.array(car_poly.exterior.coords, dtype=np.int32)
            cv2.polylines(vis_image, [coords], True, (0, 255, 0), 2)  # Green outline only
        
        # Draw slots: occupied=green fill, empty=blue fill
        for slot in slots:
            poly = slot['polygon']
            coords = np.array(list(poly.exterior.coords), dtype=np.int32)
            color = (0, 200, 0) if slot['is_occupied'] else (255, 0, 0)  # Green if occupied, Blue if empty
            cv2.fillPoly(vis_image, [coords], color)
            cv2.polylines(vis_image, [coords], True, (255, 255, 255), 1)  # White border
        
        # Calculate slot-based counts
        total_slots = len(slots)
        occupied_slots = sum(1 for s in slots if s['is_occupied'])
        empty_slots = total_slots - occupied_slots
        
        # Log detailed counts
        logger.info(f"Total slots: {total_slots} | Occupied: {occupied_slots} | Empty: {empty_slots}")
        
        # Add statistics text overlay
        cv2.putText(vis_image, f"Total Slots: {total_slots}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_image, f"Occupied: {occupied_slots}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Empty: {empty_slots}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"✓ Visualization saved to: {output_path}")
        
        # Display if requested
        if show_plot:
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Parking Slot Detection Results\n"
                     f"Total: {total_slots} | Occupied: {occupied_slots} | Empty: {empty_slots}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        logger.info("✓ Visualization complete")
        
        return vis_image
    
    def run_complete_pipeline(self, image_path, epsilon_factor=0.02, 
                              visualize_steps=False, output_dir='pipeline_outputs',
                              valid_column_centers=None):
        """
        Run the complete slot detection pipeline.
        
        Args:
            image_path: Path to input parking lot image
            epsilon_factor: Corner detection sensitivity for perspective correction
            visualize_steps: Whether to save intermediate visualizations
            output_dir: Directory for all outputs (a subdirectory with image name will be created)
            
        Returns:
            dict: Results containing all pipeline outputs
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING COMPLETE PARKING SLOT DETECTION PIPELINE")
        logger.info("="*70)
        logger.info(f"Input image: {image_path}")
        logger.info(f"Mode: {'Top-down transformation' if self.use_top_down else 'Original image (no transformation)'}")
        
        # Create output directory with image-specific subdirectory
        base_name = Path(image_path).stem
        image_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        try:
            if self.use_top_down:
                # STANDARD PIPELINE WITH TOP-DOWN TRANSFORMATION
                # Step 1: Get top-down view and parking lot boundary
                top_down_view, transform_matrix, corners, parking_lot_polygon = self.get_top_down_view(
                    image_path, epsilon_factor, visualize_steps, image_output_dir
                )
                
                # Step 2: Detect cars in original image (filtered by parking lot boundary)
                masks, scores, original_image = self.detect_cars_in_original(
                    image_path, parking_lot_polygon=parking_lot_polygon
                )
                
                # Save car detection visualization (red outlines on original image)
                if visualize_steps and len(masks) > 0:
                    car_detection_vis = original_image.copy()
                    for mask in masks:
                        mask_uint8 = mask.astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(car_detection_vis, contours, -1, (0, 0, 255), 2)  # Red outlines
                    car_detection_path = os.path.join(image_output_dir, "car_detection.png")
                    cv2.imwrite(car_detection_path, car_detection_vis)
                    logger.info(f"✓ Car detection visualization saved to: {car_detection_path}")
                
                # Step 3: Convert masks to polygons
                car_polygons = self.masks_to_polygons(masks)
                
                # Step 4: Transform polygons to top-down view (with boundary checking)
                warped_car_polygons = self.transform_polygons_to_top_down(
                    car_polygons, transform_matrix, top_down_view.shape
                )
                
                # Step 5: Find all slots (both occupied and empty) - outputs saved to image_output_dir
                slots = self.find_empty_slots(top_down_view, warped_car_polygons, debug_dir=image_output_dir, valid_column_centers=valid_column_centers)
                
                # Step 6: Visualize results with slot-based rendering
                vis_output_path = os.path.join(image_output_dir, "slot_detection.png")
                vis_image = self.visualize_results(
                    top_down_view, warped_car_polygons, slots, 
                    vis_output_path, show_plot=True
                )
                
                # Store corners for results
                corners_result = corners
                
            else:
                # SIMPLIFIED PIPELINE WITHOUT TOP-DOWN TRANSFORMATION
                logger.info("Skipping perspective transformation - using original image")
                
                # Load original image
                original_image = cv2.imread(str(image_path))
                if original_image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                
                # Step 1: Detect cars in original image (no filtering by parking lot boundary)
                masks, scores, original_image = self.detect_cars_in_original(
                    image_path, parking_lot_polygon=None
                )
                
                # Step 2: Convert masks to polygons (these are already in original image coordinates)
                car_polygons = self.masks_to_polygons(masks)
                
                # Step 3: Convert contours to Shapely polygons
                warped_car_polygons = []
                for polygon_pts in car_polygons:
                    coords = polygon_pts.reshape(-1, 2)
                    if len(coords) >= 3:
                        shapely_poly = Polygon(coords)
                        if shapely_poly.is_valid and not shapely_poly.is_empty:
                            warped_car_polygons.append(shapely_poly)
                
                logger.info(f"✓ Using {len(warped_car_polygons)} car polygons in original coordinates")
                
                # Step 4: Find all slots on original image
                slots = self.find_empty_slots(original_image, warped_car_polygons, debug_dir=image_output_dir, valid_column_centers=valid_column_centers)
                
                # Step 5: Visualize results on original image
                vis_output_path = os.path.join(image_output_dir, "slot_detection.png")
                vis_image = self.visualize_results(
                    original_image, warped_car_polygons, slots, 
                    vis_output_path, show_plot=True
                )
                
                # Use original image as "top-down" view for results
                top_down_view = original_image
                transform_matrix = None
                corners_result = None
            
            # Calculate slot-based statistics
            total_slots = len(slots)
            occupied_slots = sum(1 for s in slots if s['is_occupied'])
            empty_slots_count = total_slots - occupied_slots
            
            # Compile results with slot-based counts
            results = {
                'image_view': top_down_view,  # Either top-down or original
                'transform_matrix': transform_matrix,
                'corners': corners_result,
                'detected_cars': len(masks),
                'car_polygons': warped_car_polygons,
                'slots': slots,  # All slots with occupancy classification
                'empty_slots': [s for s in slots if not s['is_occupied']],  # Just empty ones
                'occupied_slots_list': [s for s in slots if s['is_occupied']],  # Just occupied ones
                'total_slots': total_slots,
                'occupied_slots': occupied_slots,
                'empty_slots_count': empty_slots_count,
                'visualization': vis_image,
                'used_top_down': self.use_top_down
            }
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Total parking slots: {results['total_slots']}")
            logger.info(f"Occupied slots: {results['occupied_slots']}")
            logger.info(f"Empty slots: {results['empty_slots_count']}")
            logger.info(f"Occupancy rate: {results['occupied_slots']/results['total_slots']*100:.1f}%")
            logger.info(f"Results saved to: {image_output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point for the slot detection pipeline"""
    parser = argparse.ArgumentParser(
        description='Complete parking slot detection pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with Swin+GLCM (default)
  python run_slot_detection_pipeline.py --image parking.jpg --car_checkpoint best_model.pth --yolo_model lot_detection.pt
  
  # Run with YOLO car detection
  python run_slot_detection_pipeline.py --image parking.jpg --yolo_model lot_detection.pt --use_yolo_detection --yolo_car_model yolov8n-seg.pt
  
  # Run without top-down transformation (use original image)
  python run_slot_detection_pipeline.py --image parking.jpg --car_checkpoint best_model.pth --yolo_model lot_detection.pt --no_top_down
  
  # With custom slot dimensions and shrink factor
  python run_slot_detection_pipeline.py --image parking.jpg --car_checkpoint best_model.pth --yolo_model lot_detection.pt --slot_width 162 --slot_height 57 --shrink_factor 0.90
  
  # Save intermediate visualizations
  python run_slot_detection_pipeline.py --image parking.jpg --car_checkpoint best_model.pth --yolo_model lot_detection.pt --visualize_steps
        """
    )
    
    # Required arguments
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input parking lot image')
    parser.add_argument('--car_checkpoint', type=str, default=r'../../car_detection_output_20251030_103528\checkpoints\latest_checkpoint.pth',
                       help='Path to Swin+GLCM car detector checkpoint (.pth file)')
    parser.add_argument('--yolo_model', type=str, default=r'../YOLO Files/lot_detection_best.pt',
                       help='Path to YOLO parking lot segmentation model')
    
    # Car detection model selection
    parser.add_argument('--use_yolo_detection', action='store_true',
                       help='Use YOLO for car detection instead of Swin+GLCM (default: False)')
    parser.add_argument('--yolo_car_model', type=str, default=r'../YOLO Files/car_detection_best.pt',
                       help='Path to YOLO car segmentation model (used when --use_yolo_detection is set)')
    
    # Perspective transformation toggle
    parser.add_argument('--no_top_down', action='store_true',
                       help='Skip perspective transformation and use original image (default: False, use top-down)')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='pipeline_outputs',
                       help='Directory for output files (default: pipeline_outputs)')
    parser.add_argument('--epsilon', type=float, default=0.02,
                       help='Corner detection sensitivity for perspective correction (default: 0.02)')
    parser.add_argument('--visualize_steps', action='store_true',
                       help='Save intermediate visualizations for each step')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for inference (default: cuda)')
    
    # Slot finder configuration
    parser.add_argument('--slot_width', type=int, default=165,
                       help='Standard parking slot width in pixels (default: 162)')
    parser.add_argument('--slot_height', type=int, default=45,
                       help='Standard parking slot height in pixels (default: 57)')
    parser.add_argument('--min_slot_area', type=int, default=6687,
                       help='Minimum slot area in pixels (default: 6687)')
    parser.add_argument('--max_slot_area', type=int, default=28657,
                       help='Maximum slot area in pixels (default: 28657)')
    parser.add_argument('--shrink_factor', type=float, default=0.80,
                       help='Factor to shrink car polygons (0.0-1.0, default: 0.80 = 20%% shrink)')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display result plots')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.image):
        logger.error(f"Input image not found: {args.image}")
        return
    
    if not os.path.exists(args.car_checkpoint):
        logger.error(f"Car detector checkpoint not found: {args.car_checkpoint}")
        return
    
    if not os.path.exists(args.yolo_model):
        logger.error(f"YOLO model not found: {args.yolo_model}")
        return
    
    # Validate car detection model paths
    if args.use_yolo_detection:
        if not os.path.exists(args.yolo_car_model):
            logger.error(f"YOLO car detection model not found: {args.yolo_car_model}")
            return
        logger.info("Using YOLO for car detection")
    else:
        if not os.path.exists(args.car_checkpoint):
            logger.error(f"Car detector checkpoint not found: {args.car_checkpoint}")
            return
        logger.info("Using Swin+GLCM for car detection")
    
    # Configure slot finder
    slot_config = {
        'standard_slot_width': args.slot_width,
        'standard_slot_height': args.slot_height,
        'min_slot_area': args.min_slot_area,
        'max_slot_area': args.max_slot_area,
        'shrink_factor': args.shrink_factor
    }
    
    try:
        # Initialize pipeline
        pipeline = SlotDetectionPipeline(
            car_detector_checkpoint=args.car_checkpoint,
            yolo_model_path=args.yolo_model,
            slot_config=slot_config,
            device=args.device,
            use_yolo_detection=args.use_yolo_detection,
            yolo_car_model_path=args.yolo_car_model if args.use_yolo_detection else None,
            use_top_down=not args.no_top_down  # Invert the flag
        )
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            image_path=args.image,
            epsilon_factor=args.epsilon,
            visualize_steps=args.visualize_steps,
            output_dir=args.output_dir
        )
        
        logger.info("\n✓ Slot detection pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
