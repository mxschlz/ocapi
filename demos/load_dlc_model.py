import os
import cv2
import tensorflow as tf
import time
from dlclive import DLCLive

# !!! IMPORTANT NOTE ON PERFORMANCE !!!
# `dlclive` is designed for REAL-TIME, frame-by-frame processing (e.g., from a webcam).
# For analyzing a pre-recorded video file like this, it is MUCH faster to use
# the standard `deeplabcut.analyze_videos` function, which processes frames in batches.
# See the `analyze_video_batch.py` example for the recommended approach for this task.

# --- Configuration ---
# 1. Path to the EXPORTED model FOLDER.
# IMPORTANT: Point to the FOLDER containing 'pose_cfg.yaml', not the .pb/.pt file itself.
# Based on your previous path, the folder is likely this:
model_folder = r"G:\Meine Ablage\PhD\data\OCAPI\ocapi-Max-2026-02-13\exported-models\DLC_ocapi_mobilenet_v2_1.0_iteration-0_shuffle-1"

# 2. Path to the video file to analyze
video_path = r"G:\Meine Ablage\PhD\data\OCAPI\all_videos_combined\SCS048_A_Video.mp4"

# --- Performance Settings ---
RESIZE_FACTOR = 0.5  # Downscale image by 50% (0.5). Huge speedup on CPU. Try 0.25 if still slow.
DISPLAY_VIDEO = True # Set to False to disable on-screen display for faster processing.
DYNAMIC_CROP = False
CROP_PIXELS_TO = 100

# --- Setup ---
# Verify the config file exists in the folder
config_check = os.path.join(model_folder, "pose_cfg.yaml")
if not os.path.exists(config_check):
    print(f"Error: 'pose_cfg.yaml' not found at: {config_check}")
    print("Please ensure 'model_folder' points to the exported model DIRECTORY.")
    exit()

print("-" * 20)
# --- GPU Check ---
# See if TensorFlow can detect your GPU. If not, check your CUDA/cuDNN installation.
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"--- GPU DETECTED: {gpu_devices[0].name} ---")
else:
    print("--- WARNING: NO GPU DETECTED, RUNNING ON CPU. ---")
print("-" * 20)

print(f"Loading model from: {model_folder}")
dlc = DLCLive(model_folder, resize=RESIZE_FACTOR, display=DISPLAY_VIDEO, dynamic=(DYNAMIC_CROP, 0.5, CROP_PIXELS_TO))

# --- Video Processing ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# --- Video Output Setup ---
output_path = video_path.replace(".mp4", "_dlc_analyzed.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
#writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Initialize inference with the first frame (optimizes engine for frame size)
ret, frame = cap.read()
if ret:
    dlc.init_inference(frame)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to start

print("Starting analysis... Press 'q' to quit.")
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 10 == 0:
        fps = frame_count / (time.time() - start_time)
        print(f"Processing frame {frame_count} | FPS: {fps:.2f}", end='\r')

    # Get pose: returns array of [x, y, probability] for each body part
    pose = dlc.get_pose(frame)
    print(pose)

    # Plot landmarks
    for keypoint in pose:
        x, y, prob = keypoint
        if prob > 0.5: # Only draw points with high confidence
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    #writer.write(frame)

cap.release()
#writer.release()
cv2.destroyAllWindows()
