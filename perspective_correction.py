"""
Parking Lot Perspective Correction Pipeline
============================================

Complete pipeline for:
1. Loading YOLO model for parking lot segmentation
2. Detecting parking lot boundaries
3. Finding corner points from segmentation masks
4. Creating perspective-corrected top-down views

The epsilon_factor is Key for corner detection:
- Start with 0.02 (2% of the perimeter)
- If green polygon has more than 4 corners, increase value (e.g., 0.03 or 0.04)
- If green polygon is too simple, decrease value (e.g., 0.01)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - no popups
import matplotlib.pyplot as plt
import torch

# Global YOLO model instance
yolo_model = None

def load_yolo_model(model_path='best.pt'):
    """
    Load YOLO model for parking lot segmentation.
    
    Args:
        model_path: Path to the YOLO model weights file
        
    Returns:
        Loaded YOLO model instance
    """
    global yolo_model
    yolo_model = YOLO(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"YOLO model loaded successfully from {model_path}")
    print(f"Using device: {device}")
    return yolo_model


def find_corners_from_mask(polygon_points, epsilon_factor=0.02):
    """
    Simplifies a polygon to find its four most prominent corners.
    
    Args:
        polygon_points: A NumPy array of points from the segmentation mask.
        epsilon_factor: A factor to determine the approximation accuracy. 
                        Higher values mean more simplification. Start with 0.02.
    
    Returns:
        A NumPy array of 4 corner points, or None if not found.
    """
    # Ensure polygon_points is a numpy array with correct dtype
    if not isinstance(polygon_points, np.ndarray):
        polygon_points = np.array(polygon_points, dtype=np.int32)
    
    # Ensure correct shape for cv2.arcLength - needs (N, 1, 2) or (N, 2)
    if polygon_points.ndim == 2 and polygon_points.shape[1] == 2:
        # Reshape to (N, 1, 2) for OpenCV
        polygon_points_cv = polygon_points.reshape(-1, 1, 2).astype(np.int32)
    else:
        polygon_points_cv = polygon_points.astype(np.int32)
    
    # 1. Calculate the perimeter of the closed contour
    perimeter = cv2.arcLength(polygon_points_cv, True)
    
    # Ensure perimeter is a scalar number
    if not isinstance(perimeter, (int, float, np.number)):
        print(f"Warning: perimeter is not a number, got type {type(perimeter)}: {perimeter}")
        # Try to convert or extract the numeric value
        if hasattr(perimeter, '__len__') and len(perimeter) > 0:
            perimeter = float(perimeter[0]) if isinstance(perimeter[0], (int, float, np.number)) else 0.0
        else:
            perimeter = 0.0
    
    perimeter = float(perimeter)  # Ensure it's a Python float
    
    # 2. Approximate the polygon with fewer vertices
    # Epsilon is the maximum distance from the contour to the approximated polygon.
    epsilon = perimeter * epsilon_factor
    approx_corners = cv2.approxPolyDP(polygon_points_cv, epsilon, True)
    
    # 3. We are looking for a quadrilateral. If we find one, we're done!
    if len(approx_corners) == 4:
        # The output of approxPolyDP is shape (4, 1, 2), so we reshape it
        return approx_corners.reshape(4, 2)
        
    # 4. If simplification didn't yield 4 corners (e.g., due to noise or shape complexity),
    # we can fall back to finding the four most extreme points of the original polygon.
    # This is a robust way to get a bounding quadrilateral.
    reshaped_points = polygon_points.reshape(-1, 2)
    
    # Find top-left (min sum), bottom-right (max sum)
    sum_coords = reshaped_points.sum(axis=1)
    tl = reshaped_points[np.argmin(sum_coords)]
    br = reshaped_points[np.argmax(sum_coords)]
    
    # Find top-right (min diff), bottom-left (max diff)
    diff_coords = np.diff(reshaped_points, axis=1)
    tr = reshaped_points[np.argmin(diff_coords)]
    bl = reshaped_points[np.argmax(diff_coords)]
    
    return np.array([tl, tr, br, bl], dtype=np.int32)


def order_points(pts):
    """
    Orders corner points in consistent order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: Array of 4 corner points
        
    Returns:
        Ordered array of corner points as float32
    """
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_parking_lot_coords(image_path, epsilon_factor=0.02, visualize=True, output_dir="pipeline_outputs"):
    """
    Runs YOLO inference, finds the four corners of the parking lot using the mask contour,
    and optionally visualizes the results.
    
    Args:
        image_path: Path to input image
        epsilon_factor: Corner detection sensitivity (0.01-0.04 typical range)
        visualize: Whether to save and display visualization
        output_dir: Directory to save visualization outputs
        
    Returns:
        NumPy array of 4 corner coordinates, or None if detection failed
    """
    global yolo_model
    
    if yolo_model is None:
        print("YOLO model not loaded. Call load_yolo_model() first.")
        return None
    
    # Run YOLO inference
    results = yolo_model(image_path)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    if results and results[0].masks is not None and len(results[0].masks) > 0:
        mask = results[0].masks[0]
        xy = mask.xy[0]  # Raw polygon coordinates
        
        # Debug: Check what we're getting
        print(f"DEBUG: xy type: {type(xy)}, shape: {xy.shape if hasattr(xy, 'shape') else 'no shape'}")
        print(f"DEBUG: xy dtype: {xy.dtype if hasattr(xy, 'dtype') else 'no dtype'}")
        
        # Convert to proper numpy array with int32 dtype
        xy_array = np.array(xy, dtype=np.int32)
        print(f"DEBUG: xy_array shape: {xy_array.shape}, dtype: {xy_array.dtype}")
        
        # Use the advanced corner finding
        corners = find_corners_from_mask(xy_array, epsilon_factor=epsilon_factor)

        # Fallback to minimum area rectangle if corner finding fails
        if corners is None:
            print("Could not find 4 corners from mask, falling back to minAreaRect.")
            rect = cv2.minAreaRect(xy)
            corners = cv2.boxPoints(rect)
        
        corners = np.int32(corners)

        # Visualization
        if visualize:
            viz_img = img.copy()
            
            # Draw the raw YOLO polygon in blue
            cv2.polylines(viz_img, [xy.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Draw the simplified/found corners as a green polygon
            cv2.polylines(viz_img, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Draw the individual corner points in red
            for i, p in enumerate(corners):
                cv2.circle(viz_img, tuple(p), 8, (0, 0, 255), -1)
                # Add corner labels
                cv2.putText(viz_img, str(i+1), tuple(p + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Save visualization
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_lot_corners_advanced.png")
            cv2.imwrite(output_path, viz_img)
            
            print(f"Parking lot detection with advanced corners saved to {output_path}")
            
            # Display if running interactively
            try:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
                plt.title("Detected Polygon (Blue) and Found Corners (Green with Red Points)")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            except:
                print("(matplotlib display not available in this environment)")
        
        return corners
    else:
        print("No parking lot detected in the image.")
        return None


def create_top_down_view(image_path, src_points, visualize=True, output_dir="pipeline_outputs"):
    """
    Creates a perspective-corrected top-down view of the parking lot.
    
    Args:
        image_path: Path to input image (or image array if already loaded)
        src_points: Array of 4 corner points from the original image
        visualize: Whether to display and save visualization
        output_dir: Directory to save output
        
    Returns:
        tuple: (warped_image, transformation_matrix) or (None, None) if failed
    """
    # Handle both image path and image array
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        img = image_path
        base_name = "parking_lot"
    
    if img is None:
        print("Failed to load image for top-down view creation")
        return None, None

    # Order the source points: top-left, top-right, bottom-right, bottom-left
    rect = order_points(src_points)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the top-down view
    dst_points = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst_points)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Visualization
    if visualize:
        try:
            plt.figure(figsize=(15, 7))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title("Top-Down View")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except:
            print("(matplotlib display not available in this environment)")

        # Save the top-down view
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_top_down.png")
        cv2.imwrite(output_path, warped)
        print(f"Top-down view saved to {output_path}")
    
    return warped, M


def process_parking_lot_image(image_path, model_path='best.pt', epsilon_factor=0.02, 
                              visualize=True, output_dir="pipeline_outputs"):
    """
    Complete pipeline: Load model, detect parking lot, create top-down view.
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLO model weights
        epsilon_factor: Corner detection sensitivity
        visualize: Whether to save and display visualizations
        output_dir: Directory for output files
        
    Returns:
        tuple: (top_down_image, transformation_matrix, corner_coords) or (None, None, None)
    """
    # Load model if not already loaded
    global yolo_model
    if yolo_model is None:
        load_yolo_model(model_path)
    
    # Step 1: Get parking lot coordinates
    print("\n=== Step 1: Detecting parking lot boundaries ===")
    parking_lot_coords = get_parking_lot_coords(image_path, epsilon_factor, visualize, output_dir)
    
    if parking_lot_coords is None:
        print("Failed to detect parking lot. Pipeline aborted.")
        return None, None, None
    
    print(f"Parking lot coordinates:\n{parking_lot_coords}")
    
    # # Step 2: Create top-down view
    print("\n=== Step 2: Creating perspective-corrected top-down view ===")
    warped, transform_matrix = create_top_down_view(image_path, parking_lot_coords, visualize, output_dir)
    
    if warped is not None:
        print(f"✓ Pipeline completed successfully!")
        print(f"  - Output size: {warped.shape[1]}x{warped.shape[0]} pixels")
        print(f"  - Outputs saved to: {output_dir}/")
    
    return warped, transform_matrix, parking_lot_coords


# Example usage
if __name__ == "__main__":
    # Example: Process a single image
    input_image = '../../split_data_car_annotations/train/images/01.10.24 01.00Pm_frame_293198_aug_0.jpg'
    epsilon_factor = 0.02   # Run complete pipeline
    top_down_view, transform_matrix, corners = process_parking_lot_image(
        image_path=input_image,
        model_path='../YOLO Files/lot_detection_best.pt',
        epsilon_factor=epsilon_factor,
        visualize=True,
        output_dir=f'pipeline_outputs/'
    )
    
    if top_down_view is not None:
        print("\n✓ Perspective correction complete!")
        print(f"  - Top-down view shape: {top_down_view.shape}")
        print(f"  - Transformation matrix:\n{transform_matrix}")
