"""
Parking Slot Detection - UNIFORM SLOT SIZING
=============================================
Calculates GLOBAL average car dimensions from ALL detected cars.
All empty slots will be the EXACT SAME SIZE based on this average.
"""

try:
    import cv2
    HAS_CV2 = True
except:
    cv2 = None
    HAS_CV2 = False

import numpy as np
from shapely.geometry import Polygon, Point
from sklearn.cluster import DBSCAN
import csv, os, logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('shapely').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def _write_rows_csv(rows_info, debug_dir: str):
    if not debug_dir or not rows_info:
        return
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "rows_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id","rw","rh","num_cars","num_slots_gen","num_empty","num_occupied"])
        for r in rows_info:
            w.writerow([
                r["row_id"], r["rw"], r["rh"], r["num_cars"],
                r["num_slots_gen"], r["num_empty"], r["num_occupied"]
            ])

class DynamicSlotFinder:
    def __init__(self, standard_slot_width=None, standard_slot_height=None, 
                 min_slot_area=None, max_slot_area=None, use_dynamic_sizing=False, 
                 shrink_factor=0.80, **kwargs):
        """
        Slot finder with configurable or dynamic slot dimensions.
        
        Args:
            standard_slot_width: Width of a parking slot in pixels (None for dynamic)
            standard_slot_height: Height of a parking slot in pixels (None for dynamic)
            min_slot_area: Minimum area to be considered a slot (None for dynamic)
            max_slot_area: Maximum area to be considered a slot (None for dynamic)
            use_dynamic_sizing: If True, calculate dimensions from detected cars
            shrink_factor: Factor to shrink car polygons (0.0-1.0, default: 0.80 = 20% shrink)
        """
        self.use_dynamic_sizing = use_dynamic_sizing
        self.shrink_factor = shrink_factor
        
        if use_dynamic_sizing:
            # Will be calculated from detected cars
            self.standard_slot_width = None
            self.standard_slot_height = None
            self.min_slot_area = None
            self.max_slot_area = None
            logger.info("DynamicSlotFinder: Dynamic slot sizing enabled (will calculate from detected cars)")
        else:
            # Use configured dimensions (with defaults if not provided)
            self.standard_slot_width = standard_slot_width if standard_slot_width is not None else 165
            self.standard_slot_height = standard_slot_height if standard_slot_height is not None else 45
            self.min_slot_area = min_slot_area if min_slot_area is not None else 6687
            self.max_slot_area = max_slot_area if max_slot_area is not None else 28657
            
            logger.info("DynamicSlotFinder: Using configured slot dimensions")
            logger.info(f"  Slot Width: {self.standard_slot_width}px")
            logger.info(f"  Slot Height: {self.standard_slot_height}px")
            logger.info(f"  Min Area: {self.min_slot_area}px²")
            logger.info(f"  Max Area: {self.max_slot_area}px²")
            logger.info(f"  Shrink Factor: {self.shrink_factor} ({(1-self.shrink_factor)*100:.0f}% shrink)")

    def _cluster_rows_by_x(self, car_polygons):
        """Cluster cars by X coordinate (vertical columns)."""
        if len(car_polygons) == 0:
            return []
        
        x_coords = np.array([[p.centroid.x] for p in car_polygons], dtype=np.float32)
        
        # Dynamic eps based on image width
        avg_x = np.mean(x_coords)
        eps = max(100, avg_x * 0.15)  # 15% of average X position
        
        labels = DBSCAN(eps=eps, min_samples=1).fit(x_coords).labels_
        
        rows = []
        for lbl in np.unique(labels):
            if lbl == -1:
                continue
            idxs = np.where(labels == lbl)[0]
            row_cars = [car_polygons[i] for i in idxs]
            rows.append({
                "idxs": idxs,
                "cars": row_cars
            })
        
        logger.info(f"Clustered {len(car_polygons)} cars into {len(rows)} columns (eps={eps:.1f})")
        return rows

    def _shrink_polygon(self, polygon, shrink_factor=0.98):
        """
        Minimal shrinkage (2%) to prevent overlaps.
        """
        centroid = polygon.centroid
        cx, cy = centroid.x, centroid.y
        coords = list(polygon.exterior.coords)
        new_coords = []
        
        for x, y in coords:
            dx = x - cx
            dy = y - cy
            new_x = cx + dx * shrink_factor
            new_y = cy + dy * shrink_factor
            new_coords.append((new_x, new_y))
        
        return Polygon(new_coords)

    def _create_slot_from_bbox(self, bbox, is_occupied):
        """Create slot polygon from bounding box."""
        minx, miny, maxx, maxy = bbox
        return {
            'polygon': Polygon([
                (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)
            ]),
            'is_occupied': is_occupied,
            'confidence': 1.0
        }

    def _split_gap_into_slots(self, gap_start, gap_end, gap_minx, gap_maxx, slot_height):
        """
        Split gap into uniform slots using GLOBAL slot_height.
        """
        gap_size = gap_end - gap_start
        
        # Calculate number of slots that fit
        num_slots = max(1, int(gap_size / slot_height))
        
        if num_slots > 1:
            logger.debug(f"    Splitting {gap_size:.1f}px gap → {num_slots} slots of {slot_height:.1f}px each")
        
        slots = []
        
        # Create uniform slots (may leave small remainder at end)
        for i in range(num_slots):
            slot_miny = gap_start + i * slot_height
            slot_maxy = gap_start + (i + 1) * slot_height
            
            # Don't exceed gap boundary
            if slot_maxy > gap_end:
                break
            
            slots.append(self._create_slot_from_bbox(
                (gap_minx, slot_miny, gap_maxx, slot_maxy),
                is_occupied=False
            ))
        
        return slots

    def find_slots(self, image, car_polygons, debug_dir=None, valid_column_centers=None):
        """
        UNIFORM SLOT FINDING ALGORITHM
        Uses configured slot dimensions or calculates dynamically from detected cars.
        """
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"Debug: {debug_dir}")
        
        H, W = image.shape[:2]
        
        # ============================================================
        # STEP 1: DETERMINE SLOT DIMENSIONS
        # ============================================================
        if self.use_dynamic_sizing:
            # Calculate from detected cars
            all_car_heights = []
            all_car_widths = []
            
            for car in car_polygons:
                bounds = car.bounds  # (minx, miny, maxx, maxy)
                height = bounds[3] - bounds[1]
                width = bounds[2] - bounds[0]
                all_car_heights.append(height)
                all_car_widths.append(width)
            
            # Calculate MEAN (average) of all cars
            global_avg_height = np.mean(all_car_heights) if all_car_heights else 65
            global_avg_width = np.mean(all_car_widths) if all_car_widths else 155
            
            GLOBAL_SLOT_HEIGHT = global_avg_height
            GLOBAL_SLOT_WIDTH = global_avg_width
            
            logger.info("="*60)
            logger.info(f"DYNAMIC SLOT SIZING (from {len(car_polygons)} cars):")
            logger.info(f"  Average Car Height: {global_avg_height:.1f}px → Slot Height")
            logger.info(f"  Average Car Width:  {global_avg_width:.1f}px → Slot Width")
            logger.info(f"  → UNIFORM SLOT SIZE: {GLOBAL_SLOT_HEIGHT:.1f}h × {GLOBAL_SLOT_WIDTH:.1f}w")
        else:
            # Use configured dimensions
            GLOBAL_SLOT_HEIGHT = self.standard_slot_height
            GLOBAL_SLOT_WIDTH = self.standard_slot_width
            
            logger.info("="*60)
            logger.info(f"USING CONFIGURED SLOT DIMENSIONS:")
            logger.info(f"  Slot Height: {GLOBAL_SLOT_HEIGHT}px")
            logger.info(f"  Slot Width: {GLOBAL_SLOT_WIDTH}px")
            logger.info(f"  Detected {len(car_polygons)} cars")
        
        # Gap threshold: 25% of slot height
        GLOBAL_MIN_GAP = GLOBAL_SLOT_HEIGHT * 0.25
        logger.info(f"  Min Gap Threshold: {GLOBAL_MIN_GAP:.1f}px")
        logger.info("="*60)
        
        # ============================================================
        # STEP 2: CLUSTER CARS INTO COLUMNS
        # ============================================================
        rows = self._cluster_rows_by_x(car_polygons)
        
        # If valid column centers are provided, filter out clusters that are not near them
        if valid_column_centers is not None and len(valid_column_centers) > 0:
            filtered_rows = []
            for row in rows:
                row_cars = row["cars"]
                if len(row_cars) == 0:
                    continue
                # Calculate average X of this column
                avg_x = np.mean([p.centroid.x for p in row_cars])
                
                # Check if this avg_x is close to any of the valid column centers (within 100px)
                is_valid = any(abs(avg_x - vc) < 100 for vc in valid_column_centers)
                
                if is_valid:
                    filtered_rows.append(row)
                else:
                    logger.info(f"Filtered out transient column at X={avg_x:.1f} (not near valid column centers: {valid_column_centers})")
            rows = filtered_rows
        
        all_slots = []
        rows_info = []
        
        # ============================================================
        # STEP 3: PROCESS EACH COLUMN
        # ============================================================
        for ridx, row in enumerate(rows):
            row_cars = row["cars"]
            num_cars = len(row_cars)
            
            logger.info(f"\nColumn {ridx}: {num_cars} cars detected")
            
            slots_for_row = []

            # Prepare car data with minimal shrinking
            car_data = []
            for car in row_cars:
                bounds = car.bounds
                center_y = (bounds[1] + bounds[3]) / 2
                
                # Apply configured shrinkage
                shrunk_polygon = self._shrink_polygon(car, shrink_factor=self.shrink_factor)
                shrunk_bounds = shrunk_polygon.bounds
                
                car_data.append({
                    'original_polygon': car,
                    'shrunk_polygon': shrunk_polygon,
                    'bounds': shrunk_bounds,
                    'center_y': center_y,
                    'miny': shrunk_bounds[1],
                    'maxy': shrunk_bounds[3],
                    'minx': shrunk_bounds[0],
                    'maxx': shrunk_bounds[2],
                    'height': shrunk_bounds[3] - shrunk_bounds[1]
                })
            
            car_data.sort(key=lambda x: x['center_y'])
            
            # Determine column bounds
            if len(car_data) > 0:
                all_miny = min([c['miny'] for c in car_data])
                all_maxy = max([c['maxy'] for c in car_data])
                
                # CRITICAL: Extend to full image height to detect ALL slots
                row_start = 0  # Start from top of image
                row_end = H    # Extend to bottom of image
                
                all_minx = min([c['minx'] for c in car_data])
                all_maxx = max([c['maxx'] for c in car_data])
                
                # Minimal width margin
                width_margin = GLOBAL_SLOT_WIDTH * 0.05
                col_minx = max(0, all_minx - width_margin)
                col_maxx = min(W, all_maxx + width_margin)
                
                logger.debug(f"  Y-range: [{row_start:.0f}, {row_end:.0f}] (FULL HEIGHT), X-range: [{col_minx:.0f}, {col_maxx:.0f}]")

                # ============================================================
                # STEP 3A: LEADING GAP (before first car)
                # ============================================================
                first_car = car_data[0]
                gap_start = row_start
                gap_end = first_car['miny']
                gap_size = gap_end - gap_start
                
                if gap_size >= GLOBAL_MIN_GAP:
                    logger.debug(f"  Leading gap: {gap_size:.1f}px")
                    leading_slots = self._split_gap_into_slots(
                        gap_start, gap_end, col_minx, col_maxx, GLOBAL_SLOT_HEIGHT
                    )
                    slots_for_row.extend(leading_slots)
                    logger.debug(f"    → Added {len(leading_slots)} empty slots")
                
                # ============================================================
                # STEP 3B: OCCUPIED SLOTS (actual cars)
                # ============================================================
                for car_info in car_data:
                    slots_for_row.append({
                        'polygon': car_info['shrunk_polygon'],
                        'is_occupied': True,
                        'confidence': 1.0
                    })
                
                # ============================================================
                # STEP 3C: BETWEEN GAPS (between consecutive cars)
                # ============================================================
                for i in range(len(car_data) - 1):
                    car1 = car_data[i]
                    car2 = car_data[i + 1]
                    
                    gap_start = car1['maxy']
                    gap_end = car2['miny']
                    gap_size = gap_end - gap_start
                    
                    if gap_size < GLOBAL_MIN_GAP:
                        continue
                    
                    logger.debug(f"  Between-gap (cars {i}↔{i+1}): {gap_size:.1f}px")
                    
                    between_slots = self._split_gap_into_slots(
                        gap_start, gap_end, col_minx, col_maxx, GLOBAL_SLOT_HEIGHT
                    )
                    slots_for_row.extend(between_slots)
                    logger.debug(f"    → Added {len(between_slots)} empty slots")
                
                # ============================================================
                # STEP 3D: TRAILING GAP (after last car)
                # ============================================================
                last_car = car_data[-1]
                gap_start = last_car['maxy']
                gap_end = row_end
                gap_size = gap_end - gap_start
                
                if gap_size >= GLOBAL_MIN_GAP:
                    logger.debug(f"  Trailing gap: {gap_size:.1f}px")
                    trailing_slots = self._split_gap_into_slots(
                        gap_start, gap_end, col_minx, col_maxx, GLOBAL_SLOT_HEIGHT
                    )
                    slots_for_row.extend(trailing_slots)
                    logger.debug(f"    → Added {len(trailing_slots)} empty slots")
                
                # Calculate row statistics
                num_occ = sum(1 for s in slots_for_row if s['is_occupied'])
                num_empty = len(slots_for_row) - num_occ
                
                logger.info(f"  Column {ridx} TOTAL: {len(slots_for_row)} slots ({num_empty} empty + {num_occ} occupied)")
                
                rows_info.append({
                    "row_id": ridx,
                    "rw": row_end - row_start,
                    "rh": col_maxx - col_minx,
                    "num_cars": num_cars,
                    "num_slots_gen": len(slots_for_row),
                    "num_empty": num_empty,
                    "num_occupied": num_occ
                })
                
                all_slots.extend(slots_for_row)

        # ============================================================
        # STEP 4: DEBUG VISUALIZATION
        # ============================================================
        if debug_dir and HAS_CV2:
            overlay = image.copy()
            
            empty_count = 0
            occupied_count = 0
            
            for s in all_slots:
                pts = np.array(list(s['polygon'].exterior.coords), dtype=np.int32)
                # BLUE for empty, GREEN for occupied
                color = (255, 0, 0) if not s['is_occupied'] else (0, 200, 0)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 1)
                
                if s['is_occupied']:
                    occupied_count += 1
                else:
                    empty_count += 1
            
            # Draw original car detections as yellow outlines
            for p in car_polygons:
                pts = np.array(list(p.exterior.coords), dtype=np.int32)
                cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
            
            cv2.imwrite(os.path.join(debug_dir, "debug_overlay.png"), overlay)
            logger.info(f"\nDebug visualization saved: {empty_count} empty (blue) + {occupied_count} occupied (green)")
            
            # Binary free space map
            binary_mask = np.zeros((H, W), dtype=np.uint8)
            for s in all_slots:
                if not s['is_occupied']:
                    pts = np.array(list(s['polygon'].exterior.coords), dtype=np.int32)
                    cv2.fillPoly(binary_mask, [pts], 255)
            
            cv2.imwrite(os.path.join(debug_dir, "free_space_map.png"), binary_mask)
            
            # Binary car prediction map
            car_mask = np.zeros((H, W), dtype=np.uint8)
            for p in car_polygons:
                pts = np.array(list(p.exterior.coords), dtype=np.int32)
                cv2.fillPoly(car_mask, [pts], 255)
            
            cv2.imwrite(os.path.join(debug_dir, "car_prediction_map.png"), car_mask)
            logger.info(f"Binary maps saved: free_space_map.png and car_prediction_map.png")

        _write_rows_csv(rows_info, debug_dir)
        
        total_occupied = sum(1 for s in all_slots if s['is_occupied'])
        total_empty = sum(1 for s in all_slots if not s['is_occupied'])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL RESULTS:")
        logger.info(f"  Columns: {len(rows)}")
        logger.info(f"  Cars Detected: {len(car_polygons)}")
        logger.info(f"  Total Slots: {len(all_slots)}")
        logger.info(f"  Occupied: {total_occupied}")
        logger.info(f"  Empty: {total_empty}")
        logger.info(f"  All empty slots are uniform: {GLOBAL_SLOT_HEIGHT:.1f}h × {GLOBAL_SLOT_WIDTH:.1f}w")
        logger.info(f"{'='*60}\n")
        
        return all_slots

def create_slot_finder(config=None):
    """Factory function."""
    if config is None:
        config = {}
    
    return DynamicSlotFinder(**config)
