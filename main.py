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

# Try to import YOLO for the yolo3d option
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO3D model will not be available.")
    print("Install with: pip install ultralytics")

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

def delta(a, b, c):
    # Determine the position of a point regarding the line determined by another two points
    return a.x * b.y + b.x * c.y + c.x * a.y - a.x * c.y - b.x * a.y - c.x * b.y

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

def process_video(video_file, export_knee, output_csv=None, direction=None, model='mp2d'):
    """Process video file for knee angle analysis"""
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
        yolo_model = YOLO('yolo11n-pose.pt')  # Using YOLO11 nano pose model
        
    mp_drawing = mp.solutions.drawing_utils

    if output_csv:
        csv_file = open(output_csv, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(['timeframe', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_knee_x', 'left_knee_y', 'left_knee_z',
                         'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_foot_index_x', 'left_foot_index_y', 'left_foot_index_z',
                         'left_heel_x', 'left_heel_y', 'left_heel_z', 'left_foot_direction', 'left_knee_angle', 'left_knee_correct',
                         'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_knee_x', 'right_knee_y', 'right_knee_z',
                         'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'right_foot_index_x', 'right_foot_index_y', 'right_foot_index_z',
                         'right_heel_x', 'right_heel_y', 'right_heel_z', 'right_foot_direction', 'right_knee_angle', 'right_knee_correct'])

    # Initialize plots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)  # Increase space between plots
    ax1.set_title(f'Left Knee Angle ({model.upper()})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (degrees)')
    ax2.set_title(f'Right Knee Angle ({model.upper()})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    left_knee_angles = []
    right_knee_angles = []
    timeframes = []

    # For 2D model - compute thigh and calf lengths
    left_thigh_length = 0
    right_thigh_length = 0
    left_calf_length = 0
    right_calf_length = 0

    frame_count = 0
    while True:
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

        timeframes.append(frame_count / fps)

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

                if output_csv and frame_count % frame_interval == 0:
                    writer.writerow([frame_count / fps,
                                     left_hip.x if left_hip else None, left_hip.y if left_hip else None, left_hip.z if left_hip else None,
                                     left_knee.x if left_knee else None, left_knee.y if left_knee else None, left_knee.z if left_knee else None,
                                     left_ankle.x if left_ankle else None, left_ankle.y if left_ankle else None, left_ankle.z if left_ankle else None,
                                     left_foot_index.x if left_foot_index else None, left_foot_index.y if left_foot_index else None, left_foot_index.z if left_foot_index else None,
                                     left_heel.x if left_heel else None, left_heel.y if left_heel else None, left_heel.z if left_heel else None,
                                     left_foot_direction, angle_left_knee, left_knee_correct,
                                     right_hip.x if right_hip else None, right_hip.y if right_hip else None, right_hip.z if right_hip else None,
                                     right_knee.x if right_knee else None, right_knee.y if right_knee else None, right_knee.z if right_knee else None,
                                     right_ankle.x if right_ankle else None, right_ankle.y if right_ankle else None, right_ankle.z if right_ankle else None,
                                     right_foot_index.x if right_foot_index else None, right_foot_index.y if right_foot_index else None, right_foot_index.z if right_foot_index else None,
                                     right_heel.x if right_heel else None, right_heel.y if right_heel else None, right_heel.z if right_heel else None,
                                     right_foot_direction, angle_right_knee, right_knee_correct])
                
        # Update plots for all models
        left_knee_angles.append(angle_left_knee)
        right_knee_angles.append(angle_right_knee)
        
        if len(left_knee_angles) > 1:
            if export_knee in ('left', 'both'):
                if left_knee_correct == 0:
                    ax1.plot(timeframes[-2:], left_knee_angles[-2:], 'r')
                else:
                    ax1.plot(timeframes[-2:], left_knee_angles[-2:], 'purple')
            else:
                ax1.plot(timeframes[-2:], left_knee_angles[-2:], 'green')
                
        if len(right_knee_angles) > 1:
            if export_knee in ('right', 'both'):
                if right_knee_correct == 0:
                    ax2.plot(timeframes[-2:], right_knee_angles[-2:], 'r')
                else:
                    ax2.plot(timeframes[-2:], right_knee_angles[-2:], 'blue')
            else:
                ax2.plot(timeframes[-2:], right_knee_angles[-2:], 'green')
        
        # Show the frame
        cv2.imshow(f"Video and Pose Estimation ({model.upper()})", frame_file)

        # Update the plot
        plt.pause(0.01)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

        frame_count += 1

    if output_csv:
        csv_file.close()

    # Release the video capture and close the window
    cap_file.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("Finished processing the video.")

def main(video_file, output_csv, export_knee, direction=None, model='mp2d'):
    """Main function with error handling"""
    try:
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
            
        if model == 'yolo2d' and not YOLO_AVAILABLE:
            raise ValueError("YOLO model requires ultralytics. Install with: pip install ultralytics")
            
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
                       choices=['mp2d', 'mp3d', 'yolo2d'],
                       default='mp2d',
                       help="Model type: mp2d for 2D MediaPipe, mp3d for 3D MediaPipe, or yolo2d for YOLO11-Pose 2D")

    args = parser.parse_args()

    main(args.video_file, args.output_csv, args.export_knee, args.direction, args.model)
