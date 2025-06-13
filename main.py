import cv2
import mediapipe as mp
import math
import argparse
import csv
import matplotlib.pyplot as plt
import threading
import queue
import os
import sys
import numpy as np
import glob
import urllib.request
import requests
import time

# Try to import YOLO for the yolo3d option
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO2D model will not be available.")
    print("Install with: pip install ultralytics")

# Try to import MMPose for the mmpose options
try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules
    import mmcv
    MMPOSE_AVAILABLE = True
    # Register all modules to enable model loading
    register_all_modules()
except ImportError:
    MMPOSE_AVAILABLE = False
    print("Warning: MMPose not available. Install with: pip install openmim && mim install mmengine && mim install 'mmcv>=2.0.1' && mim install 'mmpose>=1.1.0'")

# Custom pose connections
POSE_CONNECTIONS_BODY = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE, mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]

POSE_CONNECTIONS_KNEE = [
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE, mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
]

# YOLO keypoint indices (COCO format)
YOLO_KEYPOINT_INDICES = {
    'left_hip': 11,
    'left_knee': 13, 
    'left_ankle': 15,
    'right_hip': 12,
    'right_knee': 14,
    'right_ankle': 16
}

# MMPose keypoint indices (Halpe26 format for MMPose2D)
MMPOSE_KEYPOINT_INDICES = {
    'left_hip': 11,
    'left_knee': 13, 
    'left_ankle': 15,
    'right_hip': 12,
    'right_knee': 14,
    'right_ankle': 16,
    'left_foot_index': 20,  # left big toe
    'right_foot_index': 21  # right big toe
}

# RTMW keypoint indices (for MMPose3D)
RTMW_KEYPOINT_INDICES = {
    'left_hip': 11,
    'left_knee': 13, 
    'left_ankle': 15,
    'right_hip': 12,
    'right_knee': 14,
    'right_ankle': 16,
    'left_foot_index': 18,  # left big toe
    'right_foot_index': 20  # right big toe
}

def delta(a, b, c):
    # Determine the position of a point regarding the line determined by another two points
    return a.x * b.y + b.x * c.y + c.x * a.y - a.x * c.y - b.x * a.y - c.x * b.y

def delta_mmpose(a, b, c):
    # Determine the position of a point regarding the line determined by another two points
    # For MMPose keypoints (arrays with [x, y, confidence] or [x, y, z])
    return a[0] * b[1] + b[0] * c[1] + c[0] * a[1] - a[0] * c[1] - b[0] * a[1] - c[0] * b[1]

def distance2(a, b):
    # Calculate the Euclidean distance between two points
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

def distance(a, b):
    # Calculate the Euclidean distance between two points
    return math.sqrt(distance2(a, b))

def distance_3d(a, b):
    # Calculate the 3D Euclidean distance between two points
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def angle(a, b, c):
    # Calculate the measure of the angle B 
    # that is between two lines (AB and BC) 
    # determined by three points (A, B, C)
    # using the cosine theorem
    if distance2(a, b) * distance2(b, c) == 0:
        return 0
    return math.degrees(math.acos((distance2(a, b) + distance2(b, c) - distance2(a, c)) / (2 * distance(a, b) * distance(b, c))))

def calculate_angle_3d(a, b, c):
    """
    Calculate 3D angle between three points using 3D coordinates
    Args:
        a: hip point (3D)
        b: knee point (3D) 
        c: ankle point (3D)
    Returns:
        angle in degrees
    """
    # Calculate 3D vectors
    vec_ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    vec_bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vec_ba, vec_bc)
    magnitude_ba = np.linalg.norm(vec_ba)
    magnitude_bc = np.linalg.norm(vec_bc)
    
    # Avoid division by zero
    if magnitude_ba * magnitude_bc == 0:
        return 0
    
    # Calculate angle using dot product formula
    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1] to avoid numerical errors
    
    return math.degrees(math.acos(cos_angle))

def calculate_angle_yolo_2d(keypoints, side='left', direction='side', max_thigh=None, max_calf=None):
    """
    Calculate knee angle from YOLO keypoints using 2D method (similar to MediaPipe 2D)
    Args:
        keypoints: YOLO keypoints array [x, y, confidence] for each point
        side: 'left' or 'right'
        direction: 'side' or 'forward'
        max_thigh: maximum thigh length (for front view)
        max_calf: maximum calf length (for front view)
    Returns:
        angle in degrees
    """
    if side == 'left':
        hip_idx = YOLO_KEYPOINT_INDICES['left_hip']
        knee_idx = YOLO_KEYPOINT_INDICES['left_knee']
        ankle_idx = YOLO_KEYPOINT_INDICES['left_ankle']
    else:
        hip_idx = YOLO_KEYPOINT_INDICES['right_hip']
        knee_idx = YOLO_KEYPOINT_INDICES['right_knee']
        ankle_idx = YOLO_KEYPOINT_INDICES['right_ankle']
    
    # Extract coordinates (YOLO format: [x, y, confidence])
    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    ankle = keypoints[ankle_idx]
    
    # Check confidence scores (only use if confidence > 0.5)
    if hip[2] < 0.5 or knee[2] < 0.5 or ankle[2] < 0.5:
        return 0
    
    if direction == 'forward':
        # Project points onto vertical plane (similar to MediaPipe front view)
        proj_thigh = abs(hip[1] - knee[1])  
        proj_calf = abs(knee[1] - ankle[1])  
        
        thigh_ratio = proj_thigh / max_thigh if max_thigh and max_thigh != 0 else 0
        calf_ratio = proj_calf / max_calf if max_calf and max_calf != 0 else 0
        
        thigh_ratio = max(-1, min(1, thigh_ratio))
        calf_ratio = max(-1, min(1, calf_ratio))
        
        angle_thigh = math.degrees(math.acos(thigh_ratio)) if thigh_ratio <= 1 else 0
        angle_calf = math.degrees(math.acos(calf_ratio)) if calf_ratio <= 1 else 0
        
        return 180 - angle_thigh - angle_calf
    else:
        # Side view calculation
        thigh_len = math.sqrt((knee[0] - hip[0])**2 + (knee[1] - hip[1])**2)
        calf_len = math.sqrt((ankle[0] - knee[0])**2 + (ankle[1] - knee[1])**2)
        hip_ankle_len = math.sqrt((ankle[0] - hip[0])**2 + (ankle[1] - hip[1])**2)
        
        if thigh_len * calf_len == 0:
            return 0
            
        cos_angle = (calf_len**2 + thigh_len**2 - hip_ankle_len**2)/(2*calf_len*thigh_len)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.degrees(math.acos(cos_angle))

def calculate_angle_mmpose_2d(keypoints, side='left', direction='side', max_thigh=None, max_calf=None):
    """
    Calculate knee angle from MMPose keypoints using 2D method (similar to MediaPipe 2D)
    Args:
        keypoints: MMPose keypoints array [x, y, confidence] for each point
        side: 'left' or 'right'
        direction: 'side' or 'forward'
        max_thigh: maximum thigh length (for front view)
        max_calf: maximum calf length (for front view)
    Returns:
        angle in degrees
    """
    if side == 'left':
        hip_idx = MMPOSE_KEYPOINT_INDICES['left_hip']
        knee_idx = MMPOSE_KEYPOINT_INDICES['left_knee']
        ankle_idx = MMPOSE_KEYPOINT_INDICES['left_ankle']
    else:
        hip_idx = MMPOSE_KEYPOINT_INDICES['right_hip']
        knee_idx = MMPOSE_KEYPOINT_INDICES['right_knee']
        ankle_idx = MMPOSE_KEYPOINT_INDICES['right_ankle']
    
    # Extract coordinates (MMPose format: [x, y, confidence])
    hip = keypoints[hip_idx]
    knee = keypoints[knee_idx]
    ankle = keypoints[ankle_idx]
    
    # Check confidence scores (only use if confidence > 0.5)
    if hip[2] < 0.5 or knee[2] < 0.5 or ankle[2] < 0.5:
        return 0
    
    if direction == 'forward':
        # Project points onto vertical plane (similar to MediaPipe front view)
        proj_thigh = abs(hip[1] - knee[1])  
        proj_calf = abs(knee[1] - ankle[1])  
        
        thigh_ratio = proj_thigh / max_thigh if max_thigh and max_thigh != 0 else 0
        calf_ratio = proj_calf / max_calf if max_calf and max_calf != 0 else 0
        
        thigh_ratio = max(-1, min(1, thigh_ratio))
        calf_ratio = max(-1, min(1, calf_ratio))
        
        angle_thigh = math.degrees(math.acos(thigh_ratio)) if thigh_ratio <= 1 else 0
        angle_calf = math.degrees(math.acos(calf_ratio)) if calf_ratio <= 1 else 0
        
        return 180 - angle_thigh - angle_calf
    else:
        # Side view calculation
        thigh_len = math.sqrt((knee[0] - hip[0])**2 + (knee[1] - hip[1])**2)
        calf_len = math.sqrt((ankle[0] - knee[0])**2 + (ankle[1] - knee[1])**2)
        hip_ankle_len = math.sqrt((ankle[0] - hip[0])**2 + (ankle[1] - hip[1])**2)
        
        if thigh_len * calf_len == 0:
            return 0
            
        cos_angle = (calf_len**2 + thigh_len**2 - hip_ankle_len**2)/(2*calf_len*thigh_len)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.degrees(math.acos(cos_angle))

def calculate_angle_mmpose_3d(keypoints_3d, side='left'):
    """
    Calculate 3D knee angle from MMPose 3D keypoints
    Args:
        keypoints_3d: MMPose 3D keypoints array [x, y, z] for each point
        side: 'left' or 'right'
    Returns:
        angle in degrees
    """
    if side == 'left':
        hip_idx = RTMW_KEYPOINT_INDICES['left_hip']
        knee_idx = RTMW_KEYPOINT_INDICES['left_knee']
        ankle_idx = RTMW_KEYPOINT_INDICES['left_ankle']
    else:
        hip_idx = RTMW_KEYPOINT_INDICES['right_hip']
        knee_idx = RTMW_KEYPOINT_INDICES['right_knee']
        ankle_idx = RTMW_KEYPOINT_INDICES['right_ankle']
    
    # Extract 3D coordinates
    hip = keypoints_3d[hip_idx]
    knee = keypoints_3d[knee_idx]
    ankle = keypoints_3d[ankle_idx]
    
    # Calculate 2D vectors (ignoring z-coordinate for now)
    vec_knee_hip = np.array([hip[0] - knee[0], hip[1] - knee[1]])
    vec_knee_ankle = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vec_knee_hip, vec_knee_ankle)
    magnitude_knee_hip = np.linalg.norm(vec_knee_hip)
    magnitude_knee_ankle = np.linalg.norm(vec_knee_ankle)
    
    # Avoid division by zero
    if magnitude_knee_hip * magnitude_knee_ankle == 0:
        return 0
    
    # Calculate angle using dot product formula
    cos_angle = dot_product / (magnitude_knee_hip * magnitude_knee_ankle)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1] to avoid numerical errors
    
    return math.degrees(math.acos(cos_angle))

def calculate_angle(a, b, c, view_type='side', max_ab=None, max_bc=None):
    """
    Calculate angle between three points based on view type (for 2D model)
    Args:
        a: hip point
        b: knee point 
        c: ankle point
        view_type: 'side' or 'front'
        max_ab: maximum thigh length (for front view)
        max_bc: maximum calf length (for front view)
    Returns:
        angle in degrees
    """
    if view_type == 'front':
        # Project points onto vertical plane
        proj_ab = abs(a.y - b.y)  
        proj_bc = abs(b.y - c.y)  
        
        ab_ratio = proj_ab / max_ab if max_ab != 0 else 0
        bc_ratio = proj_bc / max_bc if max_bc != 0 else 0
        
        ab_ratio = max(-1, min(1, ab_ratio))
        bc_ratio = max(-1, min(1, bc_ratio))
        
        angle_ab = math.degrees(math.acos(ab_ratio))
        angle_bc = math.degrees(math.acos(bc_ratio))
        
        return 180 - angle_ab - angle_bc
    else:
        ab = math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)
        bc = math.sqrt((c.x - b.x)**2 + (c.y - b.y)**2)
        ac = math.sqrt((c.x - a.x)**2 + (c.y - a.y)**2)
        
        if ab * bc == 0:
            return 0
            
        cos_angle = (bc**2 + ab**2 - ac**2)/(2*bc*ab)
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.degrees(math.acos(cos_angle))

def download_if_missing(config_file, config_url, checkpoint_pattern, checkpoint_url):
    import glob
    # Download config if missing
    if not os.path.exists(config_file):
        print(f"[AutoDownload] Downloading config: {config_file}")
        r = requests.get(config_url)
        if r.status_code == 200:
            with open(config_file, 'w') as f:
                f.write(r.text)
        else:
            print(f"[AutoDownload] Failed to download config from {config_url}")
    # Download checkpoint if missing
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        print(f"[AutoDownload] Downloading checkpoint: {checkpoint_url}")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_url.split('/')[-1])

def export_frame_to_csv(writer, header, frame_count, fps, frame_processing_time_ms, 
                       left_hip=None, left_knee=None, left_ankle=None, left_foot_index=None,
                       right_hip=None, right_knee=None, right_ankle=None, right_foot_index=None,
                       left_foot_direction=None, angle_left_knee=None, left_knee_correct=None,
                       right_foot_direction=None, angle_right_knee=None, right_knee_correct=None,
                       export_knee='both'):
    """Export frame data to CSV using a dictionary-based approach."""
    # Build a dictionary for the row
    row_dict = {col: None for col in header}
    row_dict['frame'] = frame_count
    row_dict['timeframe'] = frame_count / fps
    row_dict['frame_processing_time_ms'] = frame_processing_time_ms

    def fill_kpt(prefix, kpt):
        if kpt is not None:
            row_dict[f'{prefix}_x'] = getattr(kpt, 'x', None)
            row_dict[f'{prefix}_y'] = getattr(kpt, 'y', None)
            row_dict[f'{prefix}_z'] = getattr(kpt, 'z', None)
            row_dict[f'{prefix}_conf'] = getattr(kpt, 'visibility', None) if hasattr(kpt, 'visibility') else (getattr(kpt, 'conf', None) if hasattr(kpt, 'conf') else None)

    # Fill in tracked knee data
    if export_knee in ('left', 'both'):
        fill_kpt('left_hip', left_hip)
        fill_kpt('left_knee', left_knee)
        fill_kpt('left_ankle', left_ankle)
        fill_kpt('left_foot_index', left_foot_index)
        row_dict['left_foot_direction'] = left_foot_direction
        row_dict['left_knee_angle'] = angle_left_knee
        row_dict['left_knee_incorrect'] = left_knee_correct

    if export_knee in ('right', 'both'):
        fill_kpt('right_hip', right_hip)
        fill_kpt('right_knee', right_knee)
        fill_kpt('right_ankle', right_ankle)
        fill_kpt('right_foot_index', right_foot_index)
        row_dict['right_foot_direction'] = right_foot_direction
        row_dict['right_knee_angle'] = angle_right_knee
        row_dict['right_knee_incorrect'] = right_knee_correct

    writer.writerow([row_dict.get(col, None) for col in header])

def process_video(video_file, export_knee, output_csv=None, direction=None, model='mp2d'):
    """Process video file and export knee angle data to CSV."""
    # Initialize video capture
    cap_file = cv2.VideoCapture(video_file)
    if not cap_file.isOpened():
        raise ValueError(f"Could not open video file: {video_file}")
    
    # Get video properties
    fps = cap_file.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
        print(f"Warning: Using default FPS value: {fps}")
    
    delay = int(1000 / fps)
    frame_interval = int(fps / 2)

    # Initialize model based on choice
    mp_pose = None
    yolo_model = None
    mmpose_model_2d = None
    mmpose_model_3d = None
    
    if model in ['mp2d', 'mp3d']:
        # Set up MediaPipe Pose with model complexity based on 2D/3D choice
        model_complexity = 2 if model == 'mp3d' else 0  # Higher complexity for 3D
        mp_pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    elif model == 'yolo2d':
        if not YOLO_AVAILABLE:
            raise ValueError("YOLO not available. Install ultralytics: pip install ultralytics")
        # Initialize YOLO pose model
        yolo_model = YOLO(os.path.join('models', 'yolo11n-pose.pt'))  # Using YOLO11 nano pose model
    elif model in ['mmpose2d', 'mmpose3d']:
        if not MMPOSE_AVAILABLE:
            raise ValueError("MMPose not available. Install with: pip install openmim && mim install mmengine && mim install 'mmcv>=2.0.1' && mim install 'mmpose>=1.1.0'")
        
        # Initialize MMPose models using proper configuration
        try:
            import subprocess
            import sys
            import glob
            
            # Use stable RTMPose models that are known to work
            if model == 'mmpose2d':
                # Use RTMPose-l for 2D wholebody pose estimation (COCO-WholeBody, 133 keypoints, 256x192)
                config_name = 'rtmpose-l_8xb512-700e_body8-halpe26-256x192'
                checkpoint_pattern = 'rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-*.pth'
                print("Using RTMPose-l for 2D wholebody pose estimation (COCO-WholeBody, 133 keypoints, 256x192)...")
                config_file = os.path.join('models', f'{config_name}.py')
                print(f"[MMPose2D] Config: {config_file}, Checkpoint pattern: {checkpoint_pattern}")
                if not os.path.exists(config_file):
                    raise ValueError(
                        f"Configuration file {config_file} not found. Please download it manually using:\n"
                        f"mim download mmpose --config {config_name} --dest models"
                    )
                checkpoint_files = glob.glob(os.path.join('models', checkpoint_pattern))
                if not checkpoint_files:
                    raise ValueError(
                        f"No checkpoint file found matching pattern {checkpoint_pattern}. "
                        "Please download the model checkpoint manually using:\n"
                        f"mim download mmpose --config {config_name} --dest models"
                    )
                checkpoint_file = checkpoint_files[0]
                device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
                mmpose_model_2d = init_model(config_file, checkpoint_file, device=device)
            elif model == 'mmpose3d':
                # Use RTMW-l for 3D pose estimation
                config_name = 'rtmw-l_8xb1024-270e_cocktail14-256x192'
                checkpoint_pattern = 'rtmw-dw-x-l_simcc-cocktail14_270e-256x192-*.pth'
                print("Using RTMW-l (Real-Time Whole-body) for 3D pose estimation...")
                config_file = os.path.join('models', f'{config_name}.py')
                print(f"[MMPose3D] Config: {config_file}, Checkpoint pattern: {checkpoint_pattern}")
                if not os.path.exists(config_file):
                    raise ValueError(
                        f"Configuration file {config_file} not found. Please download it manually using:\n"
                        f"mim download mmpose --config {config_name} --dest models"
                    )
                checkpoint_files = glob.glob(os.path.join('models', checkpoint_pattern))
                if not checkpoint_files:
                    raise ValueError(
                        f"No checkpoint file found matching pattern {checkpoint_pattern}. "
                        "Please download the model checkpoint manually using:\n"
                        f"mim download mmpose --config {config_name} --dest models"
                    )
                checkpoint_file = checkpoint_files[0]
                device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
                mmpose_model_2d = init_model(config_file, checkpoint_file, device=device)
            
        except Exception as e:
            print(f"MMPose initialization failed: {e}")
            # Final fallback: disable MMPose and suggest using other models
            raise ValueError(f"Failed to initialize MMPose model: {e}. Please try using --model mp2d, mp3d, or yolo2d instead.")

    mp_drawing = mp.solutions.drawing_utils

    if output_csv:
        csv_file = open(output_csv, mode='w', newline='')
        writer = csv.writer(csv_file)
        header = ['frame', 'timeframe', 'frame_processing_time_ms',
            'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_hip_conf',
            'left_knee_x', 'left_knee_y', 'left_knee_z', 'left_knee_conf',
            'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_ankle_conf',
            'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z', 'left_foot_index_conf',
            'left_foot_direction', 'left_knee_angle', 'left_knee_incorrect',
            'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_hip_conf',
            'right_knee_x', 'right_knee_y', 'right_knee_z', 'right_knee_conf',
            'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'right_ankle_conf',
            'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z', 'right_foot_index_conf',
            'right_foot_direction', 'right_knee_angle', 'right_knee_incorrect']
        writer.writerow(header)

    # Initialize variables for tracking
    frame_count = 0

    # For 2D model - compute thigh and calf lengths
    left_thigh_length = 0
    right_thigh_length = 0
    left_calf_length = 0
    right_calf_length = 0

    frame_count = 0
    while True:
        frame_processing_time_start = time.time()
        # Read frame from video file
        ret_file, frame_file = cap_file.read()

        if not ret_file:
            break

        # Initialize default values
        left_hip = left_knee = left_ankle = left_foot_index = left_heel = None
        right_hip = right_knee = right_ankle = right_foot_index = right_heel = None
        angle_left_knee = angle_right_knee = 0
        left_foot_direction = right_foot_direction = ""
        left_knee_correct = right_knee_correct = 1

        # Calculate frame processing time at the start
        frame_processing_time_ms = 0  # Initialize to 0, will be updated at the end of processing

        if model == 'yolo2d':
            # Process with YOLO
            results = yolo_model(frame_file, verbose=False)
            
            # Extract keypoints if person detected
            if results and len(results[0].boxes) > 0 and results[0].keypoints is not None:
                # Get first person's keypoints
                keypoints = results[0].keypoints.xy[0].cpu().numpy()  # [x, y] for each keypoint
                confidence = results[0].keypoints.conf[0].cpu().numpy()  # confidence for each keypoint
                
                # Combine coordinates and confidence
                kpts_with_conf = np.column_stack([keypoints, confidence])
                
                # For YOLO 2D model - compute thigh and calf lengths (similar to MediaPipe 2D)
                # Calculate maximum thigh and calf lengths for front view
                if export_knee in ('left', 'both'):
                    left_hip_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_hip']]
                    left_knee_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_knee']]
                    left_ankle_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_ankle']]
                    if left_hip_kpt[2] > 0.5 and left_knee_kpt[2] > 0.5 and left_ankle_kpt[2] > 0.5:
                        left_thigh_length = max(left_thigh_length, abs(left_hip_kpt[1] - left_knee_kpt[1]))
                        left_calf_length = max(left_calf_length, abs(left_knee_kpt[1] - left_ankle_kpt[1]))
                
                if export_knee in ('right', 'both'):
                    right_hip_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_hip']]
                    right_knee_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_knee']]
                    right_ankle_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_ankle']]
                    if right_hip_kpt[2] > 0.5 and right_knee_kpt[2] > 0.5 and right_ankle_kpt[2] > 0.5:
                        right_thigh_length = max(right_thigh_length, abs(right_hip_kpt[1] - right_knee_kpt[1]))
                        right_calf_length = max(right_calf_length, abs(right_knee_kpt[1] - right_ankle_kpt[1]))
                
                # Calculate angles and handle MediaPipe-style logic
                if export_knee in ('left', 'both'):
                    # Calculate angle using YOLO 2D method
                    angle_left_knee = calculate_angle_yolo_2d(
                        kpts_with_conf, 'left',
                        direction='forward' if direction == 'forward' else 'side',
                        max_thigh=left_thigh_length,
                        max_calf=left_calf_length
                    )
                    
                    # Get keypoint coordinates for drawing
                    left_hip_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_hip']]
                    left_knee_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_knee']]
                    left_ankle_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_ankle']]
                    
                    if left_hip_kpt[2] > 0.5 and left_knee_kpt[2] > 0.5 and left_ankle_kpt[2] > 0.5:
                        # Convert to pixel coordinates
                        left_hip_x = int(left_hip_kpt[0])
                        left_hip_y = int(left_hip_kpt[1])
                        left_knee_x = int(left_knee_kpt[0])
                        left_knee_y = int(left_knee_kpt[1])
                        left_ankle_x = int(left_ankle_kpt[0])
                        left_ankle_y = int(left_ankle_kpt[1])
                        
                        # Handle foot direction and correction (simplified for YOLO)
                        left_foot_direction = direction if direction else "forward"
                        left_knee_correct = 1  # Default to correct for YOLO
                        
                        # Determine line color
                        line_color = (255, 255, 255)  # White color for lines
                        
                        # Draw lines and dots (same style as MediaPipe)
                        cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), line_color, 2)
                        cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), line_color, 2)
                        cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (128, 0, 128), -1)  # Purple color for left knee dot
                        
                        # Add angle text overlay
                        text_left_knee = f"LEFT KNEE ({model.upper()})\nANGLE: {angle_left_knee:.2f}"
                        text_x_left = 10
                        text_y_left = frame_file.shape[0] - 150
                        for i, line in enumerate(text_left_knee.split('\n')):
                            cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)
                
                else:
                    # Non-selected left knee (same style as MediaPipe)
                    left_hip_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_hip']]
                    left_knee_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_knee']]
                    left_ankle_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['left_ankle']]
                    
                    if left_hip_kpt[2] > 0.5 and left_knee_kpt[2] > 0.5 and left_ankle_kpt[2] > 0.5:
                        # Calculate angle using YOLO 2D method
                        angle_left_knee = calculate_angle_yolo_2d(
                            kpts_with_conf, 'left',
                            direction='forward' if direction == 'forward' else 'side',
                            max_thigh=left_thigh_length,
                            max_calf=left_calf_length
                        )
                        
                        # Convert to pixel coordinates
                        left_hip_x = int(left_hip_kpt[0])
                        left_hip_y = int(left_hip_kpt[1])
                        left_knee_x = int(left_knee_kpt[0])
                        left_knee_y = int(left_knee_kpt[1])
                        left_ankle_x = int(left_ankle_kpt[0])
                        left_ankle_y = int(left_ankle_kpt[1])
                        
                        # Draw lines and dots (same style as MediaPipe non-selected)
                        cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), (255, 255, 255), 2)
                        cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), (255, 255, 255), 2)
                        cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (0, 255, 0), -1)  # Green color for non-selected left knee dot
                        
                        # Add angle text overlay
                        text_left_knee = f"LEFT KNEE ({model.upper()})\nANGLE: {angle_left_knee:.2f}"
                        text_x_left = 10
                        text_y_left = frame_file.shape[0] - 150
                        for i, line in enumerate(text_left_knee.split('\n')):
                            cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        left_foot_direction = ""
                        left_knee_correct = 1
                
                if export_knee in ('right', 'both'):
                    # Calculate angle using YOLO 2D method
                    angle_right_knee = calculate_angle_yolo_2d(
                        kpts_with_conf, 'right',
                        direction='forward' if direction == 'forward' else 'side',
                        max_thigh=right_thigh_length,
                        max_calf=right_calf_length
                    )
                    
                    # Get keypoint coordinates for drawing
                    right_hip_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_hip']]
                    right_knee_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_knee']]
                    right_ankle_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_ankle']]
                    
                    if right_hip_kpt[2] > 0.5 and right_knee_kpt[2] > 0.5 and right_ankle_kpt[2] > 0.5:
                        # Convert to pixel coordinates
                        right_hip_x = int(right_hip_kpt[0])
                        right_hip_y = int(right_hip_kpt[1])
                        right_knee_x = int(right_knee_kpt[0])
                        right_knee_y = int(right_knee_kpt[1])
                        right_ankle_x = int(right_ankle_kpt[0])
                        right_ankle_y = int(right_ankle_kpt[1])
                        
                        # Handle foot direction and correction (simplified for YOLO)
                        right_foot_direction = direction if direction else "forward"
                        right_knee_correct = 1  # Default to correct for YOLO
                        
                        # Determine line color
                        line_color = (255, 255, 255)  # White color for lines
                        
                        # Draw lines and dots (same style as MediaPipe)
                        cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), line_color, 2)
                        cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), line_color, 2)
                        cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (255, 0, 0), -1)  # Blue color for right knee dot
                        
                        # Add angle text overlay
                        text_right_knee = f"RIGHT KNEE ({model.upper()})\nANGLE: {angle_right_knee:.2f}"
                        text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_x_right = frame_file.shape[1] - text_size[0] - 10
                        text_y_right = frame_file.shape[0] - 150
                        for i, line in enumerate(text_right_knee.split('\n')):
                            cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                else:
                    # Non-selected right knee (same style as MediaPipe)
                    right_hip_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_hip']]
                    right_knee_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_knee']]
                    right_ankle_kpt = kpts_with_conf[YOLO_KEYPOINT_INDICES['right_ankle']]
                    
                    if right_hip_kpt[2] > 0.5 and right_knee_kpt[2] > 0.5 and right_ankle_kpt[2] > 0.5:
                        # Calculate angle using YOLO 2D method
                        angle_right_knee = calculate_angle_yolo_2d(
                            kpts_with_conf, 'right',
                            direction='forward' if direction == 'forward' else 'side',
                            max_thigh=right_thigh_length,
                            max_calf=right_calf_length
                        )
                        
                        # Convert to pixel coordinates
                        right_hip_x = int(right_hip_kpt[0])
                        right_hip_y = int(right_hip_kpt[1])
                        right_knee_x = int(right_knee_kpt[0])
                        right_knee_y = int(right_knee_kpt[1])
                        right_ankle_x = int(right_ankle_kpt[0])
                        right_ankle_y = int(right_ankle_kpt[1])
                        
                        # Draw lines and dots (same style as MediaPipe non-selected)
                        cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), (255, 255, 255), 2)
                        cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), (255, 255, 255), 2)
                        cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (0, 255, 0), -1)  # Green color for non-selected right knee dot
                        
                        # Add angle text overlay
                        text_right_knee = f"RIGHT KNEE ({model.upper()})\nANGLE: {angle_right_knee:.2f}"
                        text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        text_x_right = frame_file.shape[1] - text_size[0] - 10
                        text_y_right = frame_file.shape[0] - 150
                        for i, line in enumerate(text_right_knee.split('\n')):
                            cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        right_foot_direction = ""
                        right_knee_correct = 1

        elif model in ['mmpose2d', 'mmpose3d']:
            # Process with MMPose
            try:
                # Convert BGR to RGB for MMPose
                img_rgb = cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB)
                
                # Create a bounding box for the whole image (top-down approach)
                h, w = img_rgb.shape[:2]
                bbox = np.array([[0, 0, w, h]], dtype=np.float32)  # [x1, y1, x2, y2]
                
                # Perform inference with proper bbox format
                pose_results = inference_topdown(mmpose_model_2d, img_rgb, bbox, bbox_format='xyxy')
                
                # Default line color for MMPose
                line_color = (255, 255, 255)  # White color for lines
                
                # Extract keypoints - handle MMPose v1.x format
                keypoints = None
                keypoint_scores = None
                
                if pose_results and len(pose_results) > 0:
                    result = pose_results[0]
                    
                    # MMPose v1.x uses pred_instances
                    if hasattr(result, 'pred_instances'):
                        pred_instances = result.pred_instances
                        
                        if hasattr(pred_instances, 'keypoints') and len(pred_instances.keypoints) > 0:
                            # Convert tensor to numpy if needed
                            if hasattr(pred_instances.keypoints, 'cpu'):
                                keypoints = pred_instances.keypoints.cpu().numpy()[0]  # First person
                            else:
                                keypoints = pred_instances.keypoints[0]
                        
                        if hasattr(pred_instances, 'keypoint_scores') and len(pred_instances.keypoint_scores) > 0:
                            # Convert tensor to numpy if needed
                            if hasattr(pred_instances.keypoint_scores, 'cpu'):
                                keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()[0]  # First person
                            else:
                                keypoint_scores = pred_instances.keypoint_scores[0]
                
                # Process keypoints if found
                if keypoints is not None and len(keypoints) > 0:
                    # Combine keypoints and scores into the expected format [x, y, confidence]
                    if keypoint_scores is not None:
                        keypoints_with_conf = np.column_stack([keypoints, keypoint_scores])
                    else:
                        # Add default confidence scores if missing
                        confidence = np.ones((keypoints.shape[0], 1)) * 0.9
                        keypoints_with_conf = np.concatenate([keypoints, confidence], axis=1)
                    
                    # For MMPose 2D model - compute thigh and calf lengths
                    # Calculate maximum thigh and calf lengths for front view
                    if model == 'mmpose3d':
                        # Use RTMW keypoint indices for 3D model
                        left_hip_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['left_hip']]
                        left_knee_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['left_knee']]
                        left_ankle_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['left_ankle']]
                        right_hip_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['right_hip']]
                        right_knee_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['right_knee']]
                        right_ankle_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['right_ankle']]
                    
                        # Create Point objects for CSV export
                        if export_knee in ('left', 'both'):
                            left_hip = type('Point', (), {'x': left_hip_kpt[0], 'y': left_hip_kpt[1], 'z': 0, 'visibility': left_hip_kpt[2]})
                            left_knee = type('Point', (), {'x': left_knee_kpt[0], 'y': left_knee_kpt[1], 'z': 0, 'visibility': left_knee_kpt[2]})
                            left_ankle = type('Point', (), {'x': left_ankle_kpt[0], 'y': left_ankle_kpt[1], 'z': 0, 'visibility': left_ankle_kpt[2]})
                            try:
                                left_foot_index_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['left_foot_index']]
                                left_foot_index = type('Point', (), {'x': left_foot_index_kpt[0], 'y': left_foot_index_kpt[1], 'z': 0, 'visibility': left_foot_index_kpt[2]})
                            except:
                                left_foot_index = None
                        
                        if export_knee in ('right', 'both'):
                            right_hip = type('Point', (), {'x': right_hip_kpt[0], 'y': right_hip_kpt[1], 'z': 0, 'visibility': right_hip_kpt[2]})
                            right_knee = type('Point', (), {'x': right_knee_kpt[0], 'y': right_knee_kpt[1], 'z': 0, 'visibility': right_knee_kpt[2]})
                            right_ankle = type('Point', (), {'x': right_ankle_kpt[0], 'y': right_ankle_kpt[1], 'z': 0, 'visibility': right_ankle_kpt[2]})
                            try:
                                right_foot_index_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['right_foot_index']]
                                right_foot_index = type('Point', (), {'x': right_foot_index_kpt[0], 'y': right_foot_index_kpt[1], 'z': 0, 'visibility': right_foot_index_kpt[2]})
                            except:
                                right_foot_index = None
                    
                    else:
                        # Use Halpe26 keypoint indices for 2D model
                        left_hip_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['left_hip']]
                        left_knee_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['left_knee']]
                        left_ankle_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['left_ankle']]
                        right_hip_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['right_hip']]
                        right_knee_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['right_knee']]
                        right_ankle_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['right_ankle']]
                    
                        # Create Point objects for CSV export
                        if export_knee in ('left', 'both'):
                            left_hip = type('Point', (), {'x': left_hip_kpt[0], 'y': left_hip_kpt[1], 'z': 0, 'visibility': left_hip_kpt[2]})
                            left_knee = type('Point', (), {'x': left_knee_kpt[0], 'y': left_knee_kpt[1], 'z': 0, 'visibility': left_knee_kpt[2]})
                            left_ankle = type('Point', (), {'x': left_ankle_kpt[0], 'y': left_ankle_kpt[1], 'z': 0, 'visibility': left_ankle_kpt[2]})
                            try:
                                left_foot_index_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['left_foot_index']]
                                left_foot_index = type('Point', (), {'x': left_foot_index_kpt[0], 'y': left_foot_index_kpt[1], 'z': 0, 'visibility': left_foot_index_kpt[2]})
                            except:
                                left_foot_index = None
                        
                        if export_knee in ('right', 'both'):
                            right_hip = type('Point', (), {'x': right_hip_kpt[0], 'y': right_hip_kpt[1], 'z': 0, 'visibility': right_hip_kpt[2]})
                            right_knee = type('Point', (), {'x': right_knee_kpt[0], 'y': right_knee_kpt[1], 'z': 0, 'visibility': right_knee_kpt[2]})
                            right_ankle = type('Point', (), {'x': right_ankle_kpt[0], 'y': right_ankle_kpt[1], 'z': 0, 'visibility': right_ankle_kpt[2]})
                            try:
                                right_foot_index_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['right_foot_index']]
                                right_foot_index = type('Point', (), {'x': right_foot_index_kpt[0], 'y': right_foot_index_kpt[1], 'z': 0, 'visibility': right_foot_index_kpt[2]})
                            except:
                                right_foot_index = None
                    
                    # Use reasonable confidence threshold
                    conf_threshold = 0.3
                    
                    # Update maximum lengths
                    if (left_hip_kpt[2] > conf_threshold and left_knee_kpt[2] > conf_threshold and left_ankle_kpt[2] > conf_threshold):
                        left_thigh_length = max(left_thigh_length, abs(left_hip_kpt[1] - left_knee_kpt[1]))
                        left_calf_length = max(left_calf_length, abs(left_knee_kpt[1] - left_ankle_kpt[1]))
                    
                    if (right_hip_kpt[2] > conf_threshold and right_knee_kpt[2] > conf_threshold and right_ankle_kpt[2] > conf_threshold):
                        right_thigh_length = max(right_thigh_length, abs(right_hip_kpt[1] - right_knee_kpt[1]))
                        right_calf_length = max(right_calf_length, abs(right_knee_kpt[1] - right_ankle_kpt[1]))
                    
                    # Process selected knees
                    if export_knee in ('left', 'both'):
                        # Calculate angle using MMPose method
                        if model == 'mmpose3d':
                            # For RTMW 3D models, check if we have actual 3D keypoints
                            if keypoints_with_conf.shape[1] >= 3:
                                # We have actual 3D coordinates from RTMW
                                keypoints_3d = keypoints_with_conf[:, :3]  # x, y, z
                            else:
                                # Fallback: simulate 3D by adding z-coordinate
                                keypoints_3d = np.zeros((len(keypoints_with_conf), 3))
                                keypoints_3d[:, :2] = keypoints_with_conf[:, :2]  # x, y
                                keypoints_3d[:, 2] = 0  # z = 0 (simplified)
                            angle_left_knee = calculate_angle_mmpose_3d(keypoints_3d, 'left')
                        else:
                            angle_left_knee = calculate_angle_mmpose_2d(
                                keypoints_with_conf, 'left',
                                direction='forward' if direction == 'forward' else 'side',
                                max_thigh=left_thigh_length,
                                max_calf=left_calf_length
                            )
                        
                        # Get keypoint coordinates for drawing
                        if left_hip_kpt[2] > conf_threshold and left_knee_kpt[2] > conf_threshold and left_ankle_kpt[2] > conf_threshold:
                            # Convert to pixel coordinates
                            left_hip_x = int(left_hip_kpt[0])
                            left_hip_y = int(left_hip_kpt[1])
                            left_knee_x = int(left_knee_kpt[0])
                            left_knee_y = int(left_knee_kpt[1])
                            left_ankle_x = int(left_ankle_kpt[0])
                            left_ankle_y = int(left_ankle_kpt[1])
                            
                            # Get foot index for correction logic (if available)
                            try:
                                if model == 'mmpose3d':
                                    left_foot_index_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['left_foot_index']]
                                else:
                                    left_foot_index_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['left_foot_index']]
                                left_foot_index_x = int(left_foot_index_kpt[0])
                                left_foot_index_y = int(left_foot_index_kpt[1])
                                
                                # Determine foot direction (same logic as MediaPipe)
                                if direction:
                                    left_foot_direction = direction
                                else:
                                    # Convert to normalized coordinates for comparison
                                    left_foot_index_norm_x = left_foot_index_kpt[0] / frame_file.shape[1]
                                    left_ankle_norm_x = left_ankle_kpt[0] / frame_file.shape[1]
                                    
                                    if left_foot_index_norm_x < left_ankle_norm_x:
                                        left_foot_direction = "left"
                                    elif left_foot_index_norm_x > left_ankle_norm_x:
                                        left_foot_direction = "right"
                                    else:
                                        left_foot_direction = "forward"
                                
                                # Knee correction logic (same as MediaPipe)
                                if left_foot_direction == "forward":
                                    if delta_mmpose(left_hip_kpt, left_knee_kpt, left_ankle_kpt) < 0:  # check if left knee is on the inside
                                        line_color = (0, 0, 255)  # Red color for lines if knee is incorrect
                                        left_knee_correct = 0
                                    else:
                                        line_color = (255, 255, 255)  # White color for lines
                                        left_knee_correct = 1
                                else:
                                    # Convert to normalized coordinates for comparison
                                    left_knee_norm_x = left_knee_kpt[0] / frame_file.shape[1]
                                    left_foot_index_norm_x = left_foot_index_kpt[0] / frame_file.shape[1]
                                    
                                    if (left_foot_direction == "left" and left_knee_norm_x < left_foot_index_norm_x) or \
                                       (left_foot_direction == "right" and left_knee_norm_x > left_foot_index_norm_x):
                                        line_color = (0, 0, 255)  # Red color for lines if knee is incorrect
                                        left_knee_correct = 0
                                    else:
                                        line_color = (255, 255, 255)  # White color for lines
                                        left_knee_correct = 1
                            
                            except (KeyError, IndexError):
                                left_foot_index_kpt = None  # or skip drawing
                            
                            # Draw lines and dots (same style as MediaPipe)
                            cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), line_color, 2)
                            cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), line_color, 2)
                            cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (128, 0, 128), -1)  # Purple color for left knee dot
                            if left_foot_index_kpt is not None:
                                cv2.circle(frame_file, (left_foot_index_x, left_foot_index_y), 10, (0, 0, 255), -1)  # Red color for left foot index dot
                            
                            # Add angle text overlay
                            text_left_knee = f"LEFT KNEE ({model.upper()})\nANGLE: {angle_left_knee:.2f}"
                            text_x_left = 10
                            text_y_left = frame_file.shape[0] - 150
                            for i, line in enumerate(text_left_knee.split('\n')):
                                cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)  # Purple color for left knee text
                    
                    else:
                        # Non-selected left knee
                        if left_hip_kpt[2] > conf_threshold and left_knee_kpt[2] > conf_threshold and left_ankle_kpt[2] > conf_threshold:
                            # Calculate angle using MMPose method
                            if model == 'mmpose3d':
                                # For RTMW 3D models, check if we have actual 3D keypoints
                                if keypoints_with_conf.shape[1] >= 3:
                                    # We have actual 3D coordinates from RTMW
                                    keypoints_3d = keypoints_with_conf[:, :3]  # x, y, z
                                else:
                                    # Fallback: simulate 3D by adding z-coordinate
                                    keypoints_3d = np.zeros((len(keypoints_with_conf), 3))
                                    keypoints_3d[:, :2] = keypoints_with_conf[:, :2]  # x, y
                                    keypoints_3d[:, 2] = 0  # z = 0 (simplified)
                                angle_left_knee = calculate_angle_mmpose_3d(keypoints_3d, 'left')
                            else:
                                angle_left_knee = calculate_angle_mmpose_2d(
                                    keypoints_with_conf, 'left',
                                    direction='forward' if direction == 'forward' else 'side',
                                    max_thigh=left_thigh_length,
                                    max_calf=left_calf_length
                                )
                            
                            # Convert to pixel coordinates
                            left_hip_x = int(left_hip_kpt[0])
                            left_hip_y = int(left_hip_kpt[1])
                            left_knee_x = int(left_knee_kpt[0])
                            left_knee_y = int(left_knee_kpt[1])
                            left_ankle_x = int(left_ankle_kpt[0])
                            left_ankle_y = int(left_ankle_kpt[1])
                            
                            # Draw lines and dots (same style as MediaPipe)
                            cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), line_color, 2)
                            cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), line_color, 2)
                            cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (0, 255, 0), -1)  # Green color for left knee dot
                            
                            # Add angle text overlay
                            text_left_knee = f"LEFT KNEE ({model.upper()})\nANGLE: {angle_left_knee:.2f}"
                            text_x_left = 10
                            text_y_left = frame_file.shape[0] - 150
                            for i, line in enumerate(text_left_knee.split('\n')):
                                cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            
                            left_foot_direction = ""
                            left_knee_correct = 1
                    
                    # Process right knee similarly
                    if export_knee in ('right', 'both'):
                        # Calculate angle using MMPose method
                        if model == 'mmpose3d':
                            # For RTMW 3D models, check if we have actual 3D keypoints
                            if keypoints_with_conf.shape[1] >= 3:
                                # We have actual 3D coordinates from RTMW
                                keypoints_3d = keypoints_with_conf[:, :3]  # x, y, z
                            else:
                                # Fallback: simulate 3D by adding z-coordinate
                                keypoints_3d = np.zeros((len(keypoints_with_conf), 3))
                                keypoints_3d[:, :2] = keypoints_with_conf[:, :2]
                                keypoints_3d[:, 2] = 0
                            angle_right_knee = calculate_angle_mmpose_3d(keypoints_3d, 'right')
                        else:
                            angle_right_knee = calculate_angle_mmpose_2d(
                                keypoints_with_conf, 'right',
                                direction='forward' if direction == 'forward' else 'side',
                                max_thigh=right_thigh_length,
                                max_calf=right_calf_length
                            )
                        
                        # Get keypoint coordinates for drawing
                        if right_hip_kpt[2] > conf_threshold and right_knee_kpt[2] > conf_threshold and right_ankle_kpt[2] > conf_threshold:
                            # Convert to pixel coordinates
                            right_hip_x = int(right_hip_kpt[0])
                            right_hip_y = int(right_hip_kpt[1])
                            right_knee_x = int(right_knee_kpt[0])
                            right_knee_y = int(right_knee_kpt[1])
                            right_ankle_x = int(right_ankle_kpt[0])
                            right_ankle_y = int(right_ankle_kpt[1])
                            
                            # Get foot index for correction logic (if available)
                            try:
                                if model == 'mmpose3d':
                                    right_foot_index_kpt = keypoints_with_conf[RTMW_KEYPOINT_INDICES['right_foot_index']]
                                else:
                                    right_foot_index_kpt = keypoints_with_conf[MMPOSE_KEYPOINT_INDICES['right_foot_index']]
                                right_foot_index_x = int(right_foot_index_kpt[0])
                                right_foot_index_y = int(right_foot_index_kpt[1])
                                
                                # Determine foot direction (same logic as MediaPipe)
                                if direction:
                                    right_foot_direction = direction
                                else:
                                    # Convert to normalized coordinates for comparison
                                    right_foot_index_norm_x = right_foot_index_kpt[0] / frame_file.shape[1]
                                    right_ankle_norm_x = right_ankle_kpt[0] / frame_file.shape[1]
                                    
                                    if right_foot_index_norm_x < right_ankle_norm_x:
                                        right_foot_direction = "left"
                                    elif right_foot_index_norm_x > right_ankle_norm_x:
                                        right_foot_direction = "right"
                                    else:
                                        right_foot_direction = "forward"
                                
                                # Knee correction logic (same as MediaPipe)
                                if right_foot_direction == "forward":
                                    if delta_mmpose(right_hip_kpt, right_knee_kpt, right_ankle_kpt) > 0:  # check if right knee is on the inside
                                        line_color = (0, 0, 255)  # Red color for lines if knee is incorrect
                                        right_knee_correct = 0
                                    else:
                                        line_color = (255, 255, 255)  # White color for lines
                                        right_knee_correct = 1
                                else:
                                    # Convert to normalized coordinates for comparison
                                    right_knee_norm_x = right_knee_kpt[0] / frame_file.shape[1]
                                    right_foot_index_norm_x = right_foot_index_kpt[0] / frame_file.shape[1]
                                    
                                    if (right_foot_direction == "left" and right_knee_norm_x < right_foot_index_norm_x) or \
                                       (right_foot_direction == "right" and right_knee_norm_x > right_foot_index_norm_x):
                                        line_color = (0, 0, 255)  # Red color for lines if knee is incorrect
                                        right_knee_correct = 0
                                    else:
                                        line_color = (255, 255, 255)  # White color for lines
                                        right_knee_correct = 1
                            
                            except (KeyError, IndexError):
                                right_foot_index_kpt = None  # or skip drawing
                            
                            # Draw lines (red if incorrect, white if correct)
                            cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), line_color, 2)
                            cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), line_color, 2)
                            # Always blue dot for right knee
                            cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (255, 0, 0), -1)
                            if right_foot_index_kpt is not None:
                                cv2.circle(frame_file, (right_foot_index_x, right_foot_index_y), 10, (0, 0, 255), -1)
                            # Always blue text for right knee
                            text_right_knee = f"RIGHT KNEE ({model.upper()})\nANGLE: {angle_right_knee:.2f}"
                            text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            text_x_right = frame_file.shape[1] - text_size[0] - 10
                            text_y_right = frame_file.shape[0] - 150
                            for i, line in enumerate(text_right_knee.split('\n')):
                                cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    else:
                        # Non-selected right knee
                        if right_hip_kpt[2] > conf_threshold and right_knee_kpt[2] > conf_threshold and right_ankle_kpt[2] > conf_threshold:
                            # Calculate angle using MMPose method
                            if model == 'mmpose3d':
                                # For RTMW 3D models, check if we have actual 3D keypoints
                                if keypoints_with_conf.shape[1] >= 3:
                                    # We have actual 3D coordinates from RTMW
                                    keypoints_3d = keypoints_with_conf[:, :3]  # x, y, z
                                else:
                                    # Fallback: simulate 3D by adding z-coordinate
                                    keypoints_3d = np.zeros((len(keypoints_with_conf), 3))
                                    keypoints_3d[:, :2] = keypoints_with_conf[:, :2]
                                    keypoints_3d[:, 2] = 0
                                angle_right_knee = calculate_angle_mmpose_3d(keypoints_3d, 'right')
                            else:
                                angle_right_knee = calculate_angle_mmpose_2d(
                                    keypoints_with_conf, 'right',
                                    direction='forward' if direction == 'forward' else 'side',
                                    max_thigh=right_thigh_length,
                                    max_calf=right_calf_length
                                )
                            
                            # Convert to pixel coordinates
                            right_hip_x = int(right_hip_kpt[0])
                            right_hip_y = int(right_hip_kpt[1])
                            right_knee_x = int(right_knee_kpt[0])
                            right_knee_y = int(right_knee_kpt[1])
                            right_ankle_x = int(right_ankle_kpt[0])
                            right_ankle_y = int(right_ankle_kpt[1])
                            
                            # Draw lines and dots (same style as MediaPipe)
                            cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), line_color, 2)
                            cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), line_color, 2)
                            cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (0, 255, 0), -1)  # Green color for right knee dot
                            
                            # Add angle text overlay
                            text_right_knee = f"RIGHT KNEE ({model.upper()})\nANGLE: {angle_right_knee:.2f}"
                            text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            text_x_right = frame_file.shape[1] - text_size[0] - 10
                            text_y_right = frame_file.shape[0] - 150
                            for i, line in enumerate(text_right_knee.split('\n')):
                                cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Show the frame
                    cv2.imshow(f"Video and Pose Estimation ({model.upper()})", frame_file)
                    
                    # Exit if the 'q' key is pressed
                    if cv2.waitKey(delay) & 0xFF == ord("q"):
                        break
                    
                    # Update frame processing time before export
                    frame_processing_time_ms = (time.time() - frame_processing_time_start) * 1000


            except Exception as e:
                print(f"MMPose processing error: {e}")
                continue

        else:
            # Process with MediaPipe (existing code)
            # Convert the image color space from BGR to RGB
            frame_file_rgb = cv2.cvtColor(frame_file, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = mp_pose.process(frame_file_rgb)

            # Convert the image color space back to BGR
            frame_file = cv2.cvtColor(frame_file_rgb, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                if frame_count < 3:
                    print(f"Debug Frame {frame_count}: MediaPipe detected pose landmarks!")
                if export_knee in ('left', 'both'):
                    left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    left_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                    left_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                    left_foot_index = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX]
                    left_heel = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HEEL]

                    # Calculate angle based on model type
                    if model == 'mp3d':
                        angle_left_knee = calculate_angle_3d(left_hip, left_knee, left_ankle)
                    else:
                        # Calculate the left thigh and left calf for 2D model
                        left_thigh_length = max(left_thigh_length, abs(left_hip.y - left_knee.y))
                        left_calf_length = max(left_calf_length, abs(left_knee.y - left_ankle.y))
                        
                        angle_left_knee = calculate_angle(
                            left_hip, 
                            left_knee, 
                            left_ankle, 
                            view_type='front' if direction == 'forward' else 'side',
                            max_ab=left_thigh_length,
                            max_bc=left_calf_length
                        )

                    text_left_knee = f"LEFT KNEE ({model.upper()})\nANGLE: {angle_left_knee:.2f}"
                    text_x_left = 10  # Left side of the frame
                    text_y_left = frame_file.shape[0] - 150
                    for i, line in enumerate(text_left_knee.split('\n')):
                        cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)  # Purple color for left knee text
                    
                    left_knee_x = int(left_knee.x * frame_file.shape[1])
                    left_knee_y = int(left_knee.y * frame_file.shape[0])
                    left_hip_x = int(left_hip.x * frame_file.shape[1])
                    left_hip_y = int(left_hip.y * frame_file.shape[0])
                    left_ankle_x = int(left_ankle.x * frame_file.shape[1])
                    left_ankle_y = int(left_ankle.y * frame_file.shape[0])
                    left_foot_index_x = int(left_foot_index.x * frame_file.shape[1])
                    left_foot_index_y = int(left_foot_index.y * frame_file.shape[0])
                    
                    cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (128, 0, 128), -1)  # Purple color for left knee dot
                    
                    if direction:
                        left_foot_direction = direction
                    else:
                        if left_foot_index.x < left_ankle.x:
                            left_foot_direction = "left"
                        elif left_foot_index.x > left_ankle.x:
                            left_foot_direction = "right"
                        else:
                            left_foot_direction = "forward"

                    if left_foot_direction == "forward":
                        if delta(left_hip, left_knee, left_ankle) < 0: # check if left knee is on the inside
                            line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                            left_knee_correct = 0
                        else:
                            line_color = (255, 255, 255)  # White color for lines
                            left_knee_correct = 1
                    else:
                        if left_foot_direction == "right":
                            angle_left_knee = 360 - angle_left_knee

                        if (left_foot_direction == "left" and left_knee.x < left_foot_index.x) or (left_foot_direction == "right" and left_knee.x > left_foot_index.x):
                            line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                            left_knee_correct = 0
                        else:
                            line_color = (255, 255, 255)  # White color for lines
                            left_knee_correct = 1

                    cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), line_color, 2)
                    cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), line_color, 2)
                    cv2.circle(frame_file, (left_foot_index_x, left_foot_index_y), 10, (0, 0, 255), -1)  # Red color for left foot index dot

                else:
                    # Non-selected left knee
                    left_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    left_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
                    left_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]

                    # Calculate angle based on model type
                    if model == 'mp3d':
                        angle_left_knee = calculate_angle_3d(left_hip, left_knee, left_ankle)
                    else:
                        # Calculate the left thigh and left calf for 2D model
                        left_thigh_length = max(left_thigh_length, abs(left_hip.y - left_knee.y))
                        left_calf_length = max(left_calf_length, abs(left_knee.y - left_ankle.y))
                        
                        angle_left_knee = calculate_angle(
                            left_hip, 
                            left_knee, 
                            left_ankle, 
                            view_type='front' if direction == 'forward' else 'side',
                            max_ab=left_thigh_length,
                            max_bc=left_calf_length
                        )

                    text_left_knee = f"LEFT KNEE ({model.upper()})\nANGLE: {angle_left_knee:.2f}"
                    text_x_left = 10  # Left side of the frame  
                    text_y_left = frame_file.shape[0] - 150
                    for i, line in enumerate(text_left_knee.split('\n')):
                        cv2.putText(frame_file, line, (text_x_left, text_y_left + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green color for left knee text
                    
                    left_knee_x = int(left_knee.x * frame_file.shape[1])
                    left_knee_y = int(left_knee.y * frame_file.shape[0])
                    left_hip_x = int(left_hip.x * frame_file.shape[1])
                    left_hip_y = int(left_hip.y * frame_file.shape[0])
                    left_ankle_x = int(left_ankle.x * frame_file.shape[1])
                    left_ankle_y = int(left_ankle.y * frame_file.shape[0])
                    
                    cv2.circle(frame_file, (left_knee_x, left_knee_y), 10, (0, 255, 0), -1)  # Green color for left knee dot
                    cv2.line(frame_file, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), (255, 255, 255), 2)  # White color for lines
                    cv2.line(frame_file, (left_ankle_x, left_ankle_y), (left_knee_x, left_knee_y), (255, 255, 255), 2)  # White color for lines
                    left_foot_direction = ""
                    left_knee_correct = 1

                if export_knee in ('right', 'both'):
                    right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                    right_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                    right_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
                    right_foot_index = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX]
                    right_heel = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HEEL]

                    # Calculate angle based on model type
                    if model == 'mp3d':
                        angle_right_knee = calculate_angle_3d(right_hip, right_knee, right_ankle)
                    else:
                        # Calculate the right thigh and right calf for 2D model
                        right_thigh_length = max(right_thigh_length, abs(right_hip.y - right_knee.y))
                        right_calf_length = max(right_calf_length, abs(right_knee.y - right_ankle.y))
                        
                        angle_right_knee = calculate_angle(
                            right_hip, 
                            right_knee, 
                            right_ankle, 
                            view_type='front' if direction == 'forward' else 'side',
                            max_ab=right_thigh_length,
                            max_bc=right_calf_length
                        )

                    text_right_knee = f"RIGHT KNEE ({model.upper()})\nANGLE: {angle_right_knee:.2f}"
                    text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x_right = frame_file.shape[1] - text_size[0] - 10  # Right side of the frame
                    text_y_right = frame_file.shape[0] - 150
                    for i, line in enumerate(text_right_knee.split('\n')):
                        cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Blue color for right knee text
                    
                    right_knee_x = int(right_knee.x * frame_file.shape[1])
                    right_knee_y = int(right_knee.y * frame_file.shape[0])
                    right_hip_x = int(right_hip.x * frame_file.shape[1])
                    right_hip_y = int(right_hip.y * frame_file.shape[0])
                    right_ankle_x = int(right_ankle.x * frame_file.shape[1])
                    right_ankle_y = int(right_ankle.y * frame_file.shape[0])
                    right_foot_index_x = int(right_foot_index.x * frame_file.shape[1])
                    right_foot_index_y = int(right_foot_index.y * frame_file.shape[0])
                    
                    cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (255, 0, 0), -1)  # Blue color for right knee dot
                    
                    if direction:
                        right_foot_direction = direction
                    else:
                        if right_foot_index.x < right_ankle.x:
                            right_foot_direction = "left"
                        elif right_foot_index.x > right_ankle.x:
                            right_foot_direction = "right"
                        else:
                            right_foot_direction = "forward"

                    if right_foot_direction == "forward":
                        if delta(right_hip, right_knee, right_ankle) > 0: # check if right knee is on the inside
                            line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                            right_knee_correct = 0
                        else:
                            line_color = (255, 255, 255)  # White color for lines
                            right_knee_correct = 1
                    else:

                        if (right_foot_direction == "left" and right_knee.x < right_foot_index.x) or (right_foot_direction == "right" and right_knee.x > right_foot_index.x):
                            line_color = (0, 0, 255)  # Red color for lines if knee is in front of foot index
                            right_knee_correct = 0
                        else:
                            line_color = (255, 255, 255)  # White color for lines
                            right_knee_correct = 1

                    cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), line_color, 2)
                    cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), line_color, 2)
                    cv2.circle(frame_file, (right_foot_index_x, right_foot_index_y), 10, (0, 0, 255), -1)  # Red color for right foot index dot

                else:
                    # Non-selected right knee
                    right_hip = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                    right_knee = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
                    right_ankle = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]

                    # Calculate angle based on model type
                    if model == 'mp3d':
                        angle_right_knee = calculate_angle_3d(right_hip, right_knee, right_ankle)
                    else:
                        # Calculate the right thigh and right calf for 2D model
                        right_thigh_length = max(right_thigh_length, abs(right_hip.y - right_knee.y))
                        right_calf_length = max(right_calf_length, abs(right_knee.y - right_ankle.y))
                        
                        angle_right_knee = calculate_angle(
                            right_hip, 
                            right_knee, 
                            right_ankle, 
                            view_type='front' if direction == 'forward' else 'side',
                            max_ab=right_thigh_length,
                            max_bc=right_calf_length
                        )

                    text_right_knee = f"RIGHT KNEE ({model.upper()})\nANGLE: {angle_right_knee:.2f}"
                    text_size, _ = cv2.getTextSize(text_right_knee.split('\n')[1], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x_right = frame_file.shape[1] - text_size[0] - 10  # Right side of the frame
                    text_y_right = frame_file.shape[0] - 150
                    for i, line in enumerate(text_right_knee.split('\n')):
                        cv2.putText(frame_file, line, (text_x_right, text_y_right + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green color for right knee text
                    
                    right_knee_x = int(right_knee.x * frame_file.shape[1])
                    right_knee_y = int(right_knee.y * frame_file.shape[0])
                    right_hip_x = int(right_hip.x * frame_file.shape[1])
                    right_hip_y = int(right_hip.y * frame_file.shape[0])
                    right_ankle_x = int(right_ankle.x * frame_file.shape[1])
                    right_ankle_y = int(right_ankle.y * frame_file.shape[0])
                    
                    cv2.circle(frame_file, (right_knee_x, right_knee_y), 10, (0, 255, 0), -1)  # Green color for right knee dot
                    cv2.line(frame_file, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), (255, 255, 255), 2)  # White color for lines
                    cv2.line(frame_file, (right_ankle_x, right_ankle_y), (right_knee_x, right_knee_y), (255, 255, 255), 2)  # White color for lines
                    right_foot_direction = ""
                    right_knee_correct = 1

        # Show the frame
        cv2.imshow(f"Video and Pose Estimation ({model.upper()})", frame_file)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        # Update frame processing time before export
        frame_processing_time_ms = (time.time() - frame_processing_time_start) * 1000

        # Export to CSV if requested
        if output_csv:
            export_frame_to_csv(
                writer=writer,
                header=header,
                frame_count=frame_count,
                fps=fps,
                frame_processing_time_ms=frame_processing_time_ms,
                left_hip=left_hip if 'left_hip' in locals() else None,
                left_knee=left_knee if 'left_knee' in locals() else None,
                left_ankle=left_ankle if 'left_ankle' in locals() else None,
                left_foot_index=left_foot_index if 'left_foot_index' in locals() else None,
                right_hip=right_hip if 'right_hip' in locals() else None,
                right_knee=right_knee if 'right_knee' in locals() else None,
                right_ankle=right_ankle if 'right_ankle' in locals() else None,
                right_foot_index=right_foot_index if 'right_foot_index' in locals() else None,
                left_foot_direction=left_foot_direction if 'left_foot_direction' in locals() else None,
                angle_left_knee=angle_left_knee if 'angle_left_knee' in locals() else None,
                left_knee_correct=left_knee_correct if 'left_knee_correct' in locals() else None,
                right_foot_direction=right_foot_direction if 'right_foot_direction' in locals() else None,
                angle_right_knee=angle_right_knee if 'angle_right_knee' in locals() else None,
                right_knee_correct=right_knee_correct if 'right_knee_correct' in locals() else None,
                export_knee=export_knee
            )

        frame_count += 1

    # Clean up
    cap_file.release()
    cv2.destroyAllWindows()
    if output_csv:
        csv_file.close()
        print(f"Results saved to: {output_csv}")
    
    print("Finished processing the video.")

def main(video_file, output_csv, export_knee, direction=None, model='mp2d'):
    """Main function with error handling"""
    try:
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
            
        if model == 'yolo2d' and not YOLO_AVAILABLE:
            raise ValueError("YOLO model requires ultralytics. Install with: pip install ultralytics")
            
        if model in ['mmpose2d', 'mmpose3d'] and not MMPOSE_AVAILABLE:
            raise ValueError("MMPose models require MMPose. Install with: pip install openmim && mim install mmengine && mim install 'mmcv>=2.0.1' && mim install 'mmpose>=1.1.0'")
            
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        process_video(video_file, export_knee, output_csv, direction, model)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for knee angle analysis.")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file")
    parser.add_argument("--export_knee", type=str, 
                       choices=['left', 'right', 'both'], 
                       default='both', 
                       help="Which knee angle(s) to export")
    parser.add_argument("--direction", type=str, 
                       choices=['left', 'right', 'forward'], 
                       required=True,
                       help="Movement direction (determines view mode)")
    parser.add_argument("--model", type=str,
                       choices=['mp2d', 'mp3d', 'yolo2d', 'mmpose2d', 'mmpose3d'],
                       default='mp2d',
                       help="Model type: mp2d for 2D MediaPipe, mp3d for 3D MediaPipe, yolo2d for YOLO11-Pose 2D, mmpose2d for MMPose 2D, or mmpose3d for MMPose 3D")

    args = parser.parse_args()

    main(args.video_file, args.output_csv, args.export_knee, args.direction, args.model)
