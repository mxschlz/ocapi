import deeplabcut
import os
os.environ["QT_API"] = "pyqt5"
import yaml
from pathlib import Path
import re

# --- IMPORTANT ---
# This script uses the standard, high-performance DeepLabCut batch analysis function.
# It is the RECOMMENDED way to analyze pre-recorded videos for maximum speed.

# --- Configuration ---
# 1. Load the OCAPI config to automatically find the DeepLabCut project config.
try:
    ocapi_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.yml"))
except NameError:
    # Fallback if __file__ is not defined (e.g. interactive console)
    if os.path.exists("config.yml"):
        ocapi_config_path = os.path.abspath("config.yml")
    elif os.path.exists("../config.yml"):
        ocapi_config_path = os.path.abspath("../config.yml")
    else:
        ocapi_config_path = r"C:/Users/Max/PycharmProjects/ocapi/config.yml"

if not os.path.exists(ocapi_config_path):
    raise FileNotFoundError(f"OCAPI config not found at: {ocapi_config_path}")

with open(ocapi_config_path, 'r') as f:
    ocapi_conf = yaml.safe_load(f)

# Extract the DLC model path defined in config.yml
dlc_model_folder = ocapi_conf.get("Features", {}).get("DLC_MODEL_PATH", "")
if not dlc_model_folder:
    raise ValueError("DLC_MODEL_PATH is not defined in config.yml")

# Clean up potential python-style raw string formatting from YAML (e.g. r"path")
dlc_model_folder = dlc_model_folder.strip().lstrip('r').strip('"\'')

# Derive the path to the main DeepLabCut project config.yaml
# We assume the structure: project_root/exported-models/model_folder
# So we go up two levels from the exported model folder to find the project root.
path_config_file = str(Path(dlc_model_folder).parents[1] / "config.yaml")

# 2. Path to the video file you want to analyze.
video_path = r"G:\Meine Ablage\PhD\data\OCAPI\all_videos_combined\SCS048_A_Video.mp4"

# --- Verification ---
if not os.path.exists(path_config_file):
    raise FileNotFoundError(f"The main project config.yaml was not found at: {path_config_file}")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"The video file was not found at: {video_path}")

# --- FIX: Ensure project_path in config.yaml matches the current location ---
# Since training was done on Linux, the project_path in config.yaml is likely a Linux path.
# We must update it to the current Windows path for DLC to find the training data.
# FORCE FORWARD SLASHES to avoid YAML escape sequence errors on Windows (e.g. \M in \Meine)
current_project_root = str(Path(path_config_file).parent.resolve()).replace("\\", "/")

with open(path_config_file, 'r') as f:
    config_content = f.read()

# Find the project_path line using regex to preserve comments
match = re.search(r"^project_path:\s*(.*)$", config_content, re.MULTILINE)
if match:
    current_line_in_file = match.group(0)
    old_path_str = match.group(1).strip().strip('"').strip("'")
    
    # We update if the path is factually different OR if the file uses backslashes (which breaks YAML)
    path_mismatch = os.path.normpath(old_path_str) != os.path.normpath(current_project_root)
    has_backslashes = "\\" in current_line_in_file

    if path_mismatch or has_backslashes:
        print(f"Updating project_path in config.yaml to '{current_project_root}'")
        new_line = f'project_path: "{current_project_root}"'
        new_content = config_content.replace(current_line_in_file, new_line)
        with open(path_config_file, 'w') as f:
            f.write(new_content)

# --- FIX: Extract shuffle index from the model folder name ---
shuffle_match = re.search(r"shuffle-(\d+)", dlc_model_folder)
shuffle_idx = int(shuffle_match.group(1)) if shuffle_match else 1

print("--- Starting Batch Video Analysis with DeepLabCut ---")
print(f"Using project config: {path_config_file}")
print(f"Analyzing video: {video_path}")

# --- Run Batch Analysis ---
# This function is highly optimized. It will automatically use the GPU if available
# and processes frames in batches for maximum throughput.
deeplabcut.analyze_videos(path_config_file, [video_path],
                          shuffle=shuffle_idx,
                          destfolder=r"G:\Meine Ablage\PhD\data\OCAPI\output\dlc",
                          save_as_csv=True)

print("\n--- Analysis complete! ---")
print("Check the directory where your video is located for the output .h5 and .csv files.")
