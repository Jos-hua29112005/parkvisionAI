"""
Parking Slot Detection Web Application
======================================

A Flask-based web application for detecting parking slots using
Swin Transformer + GLCM car detection model.

Features:
- Static image upload and detection
- Real-time video simulation with live updates

Run with: python app.py
Access at: http://localhost:5000
"""

# IMPORTANT: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - prevents popup windows

import os
import sys
import uuid
import json
import time
import shutil
import threading
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from queue import Queue

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
from huggingface_hub import hf_hub_download

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'parking-detection-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['VIDEO_FOLDER'] = 'videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max for videos
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)

# Global model instances (lazy loaded)
car_detector = None
pipeline = None
models_loaded = False

# Real-time video processing state
video_processor = None
realtime_active = False
realtime_results = Queue(maxsize=10)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def allowed_video(filename):
    """Check if video extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']


class VideoProcessor:
    """Process video frames for real-time parking detection simulation"""
    
    def __init__(self, video_path, frame_interval=2.0):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.cap = None
        self.running = False
        self.thread = None
        self.current_frame = None
        self.current_results = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 30
        self.lock = threading.Lock()
        
    def start(self):
        """Start video processing in background thread"""
        if self.running:
            return
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop video processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def _process_loop(self):
        """Main processing loop - runs in background thread"""
        global car_detector, pipeline, models_loaded
        
        if not models_loaded:
            load_models()
            
        frames_to_skip = int(self.fps * self.frame_interval)
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if self.frame_count % frames_to_skip != 0:
                continue
                
            try:
                results = self._detect_frame(frame)
                
                with self.lock:
                    self.current_frame = frame.copy()
                    self.current_results = results
                    
                if not realtime_results.full():
                    realtime_results.put(results)
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                
            time.sleep(0.1)
            
    def _detect_frame(self, frame):
        """Run detection on a single frame"""
        global car_detector, pipeline
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_frame_{threading.current_thread().ident}.jpg')
        cv2.imwrite(temp_path, frame)
        
        start_time = time.time()
        
        try:
            if pipeline is not None:
                temp_output = os.path.join(app.config['RESULTS_FOLDER'], f'temp_realtime_{threading.current_thread().ident}')
                os.makedirs(temp_output, exist_ok=True)
                
                results = pipeline.run_complete_pipeline(
                    image_path=temp_path,
                    epsilon_factor=0.02,
                    visualize_steps=False,
                    output_dir=temp_output
                )
                
                processing_time = time.time() - start_time
                
                # Extract slot data for visual parking map
                slots_data = []
                all_slots = results.get('slots', [])
                
                # Group slots by column (based on x-coordinate - parking has vertical columns)
                if all_slots:
                    # Extract center coordinates from polygon
                    slots_with_centers = []
                    for slot in all_slots:
                        poly = slot.get('polygon')
                        if poly:
                            # Get centroid of polygon
                            centroid = poly.centroid
                            center_x = centroid.x
                            center_y = centroid.y
                        else:
                            center_x, center_y = 0, 0
                        
                        slots_with_centers.append({
                            'slot': slot,
                            'center_x': center_x,
                            'center_y': center_y
                        })
                    
                    # Sort by x-coordinate to group into columns (vertical parking rows)
                    sorted_by_x = sorted(slots_with_centers, key=lambda s: s['center_x'])
                    
                    # Cluster into columns based on x-position
                    column_threshold = 80  # pixels between columns
                    columns = []
                    current_column = []
                    last_x = None
                    
                    for item in sorted_by_x:
                        if last_x is None or abs(item['center_x'] - last_x) <= column_threshold:
                            current_column.append(item)
                        else:
                            if current_column:
                                columns.append(current_column)
                            current_column = [item]
                        last_x = item['center_x']
                    
                    if current_column:
                        columns.append(current_column)
                    
                    # Now sort each column by y-coordinate (top to bottom)
                    slot_idx = 0
                    for col_idx, column in enumerate(columns):
                        # Sort by y within each column
                        column_sorted = sorted(column, key=lambda s: s['center_y'])
                        
                        for item in column_sorted:
                            slots_data.append({
                                'id': slot_idx,
                                'row': col_idx,  # Column becomes "row" in the visual
                                'is_occupied': item['slot'].get('is_occupied', False),
                                'center_x': int(item['center_x']),
                                'center_y': int(item['center_y']),
                            })
                            slot_idx += 1
                
                # Count rows (columns in the parking lot)
                num_rows = len(columns) if all_slots else 0
                
                return {
                    'success': True,
                    'frame_number': self.frame_count,
                    'total_frames': self.total_frames,
                    'progress': round(self.frame_count / self.total_frames * 100, 1) if self.total_frames > 0 else 0,
                    'num_cars': results.get('detected_cars', 0),
                    'total_slots': results.get('total_slots', 0),
                    'occupied_slots': results.get('occupied_slots', 0),
                    'empty_slots': results.get('empty_slots_count', 0),
                    'occupancy_rate': round(results.get('occupied_slots', 0) / results.get('total_slots', 1) * 100, 1) if results.get('total_slots', 0) > 0 else 0,
                    'processing_time': round(processing_time, 2),
                    'slots': slots_data,
                    'num_rows': num_rows,
                    'timestamp': datetime.now().isoformat()
                }
                
            elif car_detector is not None:
                output_path = os.path.join(app.config['RESULTS_FOLDER'], f'temp_detect_{threading.current_thread().ident}.png')
                vis_image, num_cars = car_detector.process_image(
                    temp_path,
                    output_path=output_path,
                    show_plot=False
                )
                
                processing_time = time.time() - start_time
                
                return {
                    'success': True,
                    'frame_number': self.frame_count,
                    'total_frames': self.total_frames,
                    'progress': round(self.frame_count / self.total_frames * 100, 1) if self.total_frames > 0 else 0,
                    'num_cars': num_cars,
                    'total_slots': 0,
                    'occupied_slots': 0,
                    'empty_slots': 0,
                    'occupancy_rate': 0,
                    'processing_time': round(processing_time, 2),
                    'slots': [],
                    'num_rows': 0,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': False,
                'error': str(e),
                'frame_number': self.frame_count,
                'total_frames': self.total_frames,
                'progress': round(self.frame_count / self.total_frames * 100, 1) if self.total_frames > 0 else 0,
                'frame_image': f'data:image/jpeg;base64,{frame_base64}',
                'timestamp': datetime.now().isoformat()
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
    def get_current_state(self):
        """Get current processing state"""
        with self.lock:
            return {
                'running': self.running,
                'frame_count': self.frame_count,
                'total_frames': self.total_frames,
                'progress': round(self.frame_count / self.total_frames * 100, 1) if self.total_frames > 0 else 0,
                'results': self.current_results
            }


def load_models():
    """Lazy load models on first request"""
    global car_detector, pipeline, models_loaded
    
    if models_loaded:
        return True
    
    try:
        from visualize_predictions import CarDetectorVisualizer
        from run_slot_detection_pipeline import SlotDetectionPipeline
        
        # Hugging Face model repository
        HF_REPO_ID = os.environ.get('HF_MODEL_REPO', 'Arvind007m/parking-slot-models')
        
        # First check if models exist locally
        car_checkpoint = 'models/car_detector.pth'
        yolo_model = 'models/lot_detector.pt'
        
        # Download from Hugging Face if not available locally
        if not os.path.exists(car_checkpoint):
            print(f"Downloading car detector from Hugging Face: {HF_REPO_ID}...")
            try:
                car_checkpoint = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="car_detector.pth",
                    cache_dir="models"
                )
            except Exception as e:
                print(f"Could not download car detector: {e}")
                return False
        
        if not os.path.exists(yolo_model):
            print(f"Downloading YOLO model from Hugging Face: {HF_REPO_ID}...")
            try:
                yolo_model = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename="lot_detector.pt",
                    cache_dir="models"
                )
            except Exception as e:
                print(f"Warning: Could not download YOLO model: {e}")
        
        # Use CPU if CUDA not available (for cloud deployment)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load car detector
        print("Loading car detection model...")
        car_detector = CarDetectorVisualizer(
            checkpoint_path=car_checkpoint,
            device=device,
            confidence_threshold=0.5
        )
        
        # Load full pipeline if YOLO model exists
        if os.path.exists(yolo_model):
            print("Loading full pipeline...")
            slot_config = {
                'standard_slot_width': 162,
                'standard_slot_height': 57,
                'min_slot_area': 6687,
                'max_slot_area': 28657,
            }
            pipeline = SlotDetectionPipeline(
                car_detector_checkpoint=car_checkpoint,
                yolo_model_path=yolo_model,
                slot_config=slot_config,
                device=device
            )
        
        models_loaded = True
        print("Models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/realtime')
def realtime():
    """Serve the real-time monitoring page"""
    return render_template('realtime.html')


@app.route('/api/status')
def api_status():
    """Check if models are loaded"""
    return jsonify({
        'models_loaded': models_loaded,
        'car_detector_available': car_detector is not None,
        'pipeline_available': pipeline is not None
    })


@app.route('/api/load-models', methods=['POST'])
def api_load_models():
    """Load models on demand"""
    success = load_models()
    return jsonify({
        'success': success,
        'models_loaded': models_loaded,
        'car_detector_available': car_detector is not None,
        'pipeline_available': pipeline is not None
    })


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """
    Run detection on uploaded image
    
    Parameters (form data):
        - file: Image file
        - mode: 'detect' (cars only) or 'full' (complete pipeline)
        - confidence: Detection confidence threshold (0.0-1.0)
    """
    start_time = time.time()
    
    # Check if models are loaded
    if not models_loaded:
        success = load_models()
        if not success:
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get parameters
    mode = request.form.get('mode', 'full')
    confidence = float(request.form.get('confidence', 0.5))
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    base_name = Path(filename).stem
    extension = Path(filename).suffix
    upload_filename = f"{base_name}_{request_id}{extension}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    file.save(upload_path)
    
    # Create results directory for this request
    results_dir = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_{request_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        if mode == 'detect':
            # Car detection only
            if car_detector is None:
                return jsonify({'error': 'Car detector not loaded'}), 500
            
            # Update confidence threshold
            car_detector.confidence_threshold = confidence
            
            # Run detection
            output_path = os.path.join(results_dir, 'detection.png')
            vis_image, num_detections = car_detector.process_image(
                upload_path,
                output_path=output_path,
                show_plot=False
            )
            
            processing_time = time.time() - start_time
            
            return jsonify({
                'success': True,
                'mode': 'detect',
                'request_id': request_id,
                'num_cars': num_detections,
                'processing_time': round(processing_time, 2),
                'results': {
                    'detection': url_for('serve_result', filepath=f"{base_name}_{request_id}/detection.png")
                }
            })
        
        else:  # Full pipeline
            if pipeline is None:
                return jsonify({'error': 'Full pipeline not available (YOLO model missing)'}), 500
            
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(
                image_path=upload_path,
                epsilon_factor=0.02,
                visualize_steps=True,
                output_dir=results_dir
            )
            
            processing_time = time.time() - start_time
            
            # Build result URLs - scan recursively for all image files
            result_files = {}
            folder_name = f"{base_name}_{request_id}"
            
            # Walk through all subdirectories to find image files
            for root, dirs, files in os.walk(results_dir):
                for fname in files:
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        # Get relative path from results_dir
                        rel_path = os.path.relpath(os.path.join(root, fname), app.config['RESULTS_FOLDER'])
                        # Use forward slashes for URL
                        rel_path = rel_path.replace('\\', '/')
                        key = Path(fname).stem
                        result_files[key] = url_for('serve_result', filepath=rel_path)
            
            return jsonify({
                'success': True,
                'mode': 'full',
                'request_id': request_id,
                'num_cars': results['detected_cars'],
                'total_slots': results['total_slots'],
                'occupied_slots': results['occupied_slots'],
                'empty_slots': results['empty_slots_count'],
                'occupancy_rate': round(results['occupied_slots'] / results['total_slots'] * 100, 1) if results['total_slots'] > 0 else 0,
                'processing_time': round(processing_time, 2),
                'results': result_files
            })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/results/<path:filepath>')
def serve_result(filepath):
    """Serve result images from nested folders"""
    # filepath can be like "test_abc123/test_abc123/slot_detection.png"
    full_path = os.path.join(app.config['RESULTS_FOLDER'], filepath)
    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============== Real-time Video Processing Endpoints ==============

@app.route('/api/realtime/upload', methods=['POST'])
def api_realtime_upload():
    """Upload a video file for real-time simulation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not allowed_video(file.filename):
        return jsonify({'error': 'Video format not supported. Use MP4, AVI, MOV, MKV, or WMV'}), 400
        
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    file.save(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.remove(video_path)
        return jsonify({'error': 'Cannot read video file'}), 400
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return jsonify({
        'success': True,
        'filename': filename,
        'path': video_path,
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'duration': round(duration, 2),
        'resolution': f'{width}x{height}'
    })


@app.route('/api/realtime/start', methods=['POST'])
def api_realtime_start():
    """Start real-time video processing"""
    global video_processor, realtime_active
    
    data = request.get_json() or {}
    video_path = data.get('video_path')
    frame_interval = float(data.get('frame_interval', 2.0))
    
    if not video_path:
        video_files = os.listdir(app.config['VIDEO_FOLDER'])
        if video_files:
            video_path = os.path.join(app.config['VIDEO_FOLDER'], video_files[0])
        else:
            return jsonify({'error': 'No video file specified or found'}), 400
            
    if not os.path.exists(video_path):
        return jsonify({'error': f'Video file not found: {video_path}'}), 404
        
    if video_processor and video_processor.running:
        video_processor.stop()
        
    if not models_loaded:
        success = load_models()
        if not success:
            return jsonify({'error': 'Failed to load detection models'}), 500
            
    try:
        video_processor = VideoProcessor(video_path, frame_interval)
        video_processor.start()
        realtime_active = True
        
        return jsonify({
            'success': True,
            'message': 'Real-time processing started',
            'video_path': video_path,
            'frame_interval': frame_interval
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/realtime/stop', methods=['POST'])
def api_realtime_stop():
    """Stop real-time video processing"""
    global video_processor, realtime_active
    
    if video_processor:
        video_processor.stop()
        video_processor = None
        
    realtime_active = False
    
    return jsonify({
        'success': True,
        'message': 'Real-time processing stopped'
    })


@app.route('/api/realtime/status')
def api_realtime_status():
    """Get current real-time processing status"""
    global video_processor, realtime_active
    
    if video_processor and video_processor.running:
        state = video_processor.get_current_state()
        return jsonify({
            'active': True,
            **state
        })
    else:
        return jsonify({
            'active': False,
            'running': False
        })


@app.route('/api/realtime/stream')
def api_realtime_stream():
    """Server-Sent Events stream for real-time updates"""
    def generate():
        global video_processor, realtime_active
        
        while realtime_active and video_processor and video_processor.running:
            try:
                if not realtime_results.empty():
                    result = realtime_results.get(timeout=0.5)
                    yield f"data: {json.dumps(result)}\n\n"
                else:
                    state = video_processor.get_current_state()
                    if state['results']:
                        yield f"data: {json.dumps(state['results'])}\n\n"
                    time.sleep(0.5)
            except Exception as e:
                time.sleep(0.5)
                
        yield f"data: {json.dumps({'stopped': True})}\n\n"
        
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/realtime/videos')
def api_realtime_videos():
    """List available videos for processing"""
    videos = []
    for filename in os.listdir(app.config['VIDEO_FOLDER']):
        if allowed_video(filename):
            filepath = os.path.join(app.config['VIDEO_FOLDER'], filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            videos.append({
                'filename': filename,
                'path': filepath,
                'size_mb': round(size_mb, 2)
            })
    return jsonify({'videos': videos})


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Check if running in production (Render sets PORT env variable)
    port = int(os.environ.get('PORT', 5000))
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    print("=" * 60)
    print("PARKING SLOT DETECTION WEB APPLICATION")
    print("=" * 60)
    print(f"\nEnvironment: {'Production' if is_production else 'Development'}")
    print(f"Starting server on port {port}...")
    print(f"Access the app at: http://localhost:{port}")
    print(f"Real-time monitor at: http://localhost:{port}/realtime")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    if is_production:
        # Production mode - gunicorn handles this, but fallback just in case
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode with reloader disabled
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
