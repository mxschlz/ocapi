import numpy as np
import socket
import time
import logging
import csv
from datetime import datetime
import datetime as dt
import os
from AngleBuffer import AngleBuffer
import cv2 as cv
import re
import mediapipe as mp
import yaml
from dataclasses import dataclass
from typing import Optional, Any
from sklearn.cluster import DBSCAN # Add this import at the top of the file


# some good aesthetics
try:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
except AttributeError:
    # MediaPipe solutions not available (likely due to protobuf conflict with DLCLive)
    mp_drawing = None
    mp_drawing_styles = None
    mp_face_mesh = None
    drawing_spec = None


@dataclass
class FaceLandmarks:
    """Standardized container for semantic face landmarks, agnostic of the detector backend."""
    left_iris_center: Optional[np.ndarray] = None
    right_iris_center: Optional[np.ndarray] = None
    left_eye_outer_corner: Optional[np.ndarray] = None
    right_eye_outer_corner: Optional[np.ndarray] = None
    
    # Points required for Head Pose Estimation (PnP)
    nose_tip: Optional[np.ndarray] = None
    chin: Optional[np.ndarray] = None
    left_mouth_corner: Optional[np.ndarray] = None
    right_mouth_corner: Optional[np.ndarray] = None
    
    # Raw data for logging or specific calculations (Backend specific)
    raw_mesh_points_2d: Optional[np.ndarray] = None
    raw_mesh_points_3d: Optional[np.ndarray] = None
    mp_multi_face_landmarks: Any = None # Keep original MP object for drawing utils


class MediaPipeDetector:
    """Encapsulates MediaPipe specific logic and indices."""
    def __init__(self, config):
        self.config = config
        if mp_face_mesh is None:
            raise ImportError("MediaPipe solutions module is not available. Check for protobuf version conflicts.")
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=config.MAX_NUM_FACES, 
            refine_landmarks=config.USE_ATTENTION_MESH,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )

    def detect(self, frame) -> Optional[FaceLandmarks]:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks_mp = results.multi_face_landmarks[0]
        img_h, img_w = frame.shape[:2]
        
        # Convert to numpy array
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in face_landmarks_mp.landmark])
        mesh_points_3d = np.array([[n.x, n.y, n.z] for n in face_landmarks_mp.landmark])

        # Extract semantic points using config indices
        # 1. Iris Centers (calculated via MinEnclosingCircle for MP)
        (l_cx, l_cy), _ = cv.minEnclosingCircle(mesh_points[self.config.LEFT_EYE_IRIS])
        (r_cx, r_cy), _ = cv.minEnclosingCircle(mesh_points[self.config.RIGHT_EYE_IRIS])
        
        return FaceLandmarks(
            left_iris_center=np.array([l_cx, l_cy]),
            right_iris_center=np.array([r_cx, r_cy]),
            left_eye_outer_corner=mesh_points[self.config.LEFT_EYE_OUTER_CORNER].astype(np.float32),
            right_eye_outer_corner=mesh_points[self.config.RIGHT_EYE_OUTER_CORNER].astype(np.float32),
            
            nose_tip=mesh_points[self.config.NOSE_TIP_INDEX].astype(np.float32),
            chin=mesh_points[self.config.CHIN_INDEX].astype(np.float32),
            left_mouth_corner=mesh_points[self.config.LEFT_MOUTH_CORNER].astype(np.float32),
            right_mouth_corner=mesh_points[self.config.RIGHT_MOUTH_CORNER].astype(np.float32),
            
            raw_mesh_points_2d=mesh_points,
            raw_mesh_points_3d=mesh_points_3d,
            mp_multi_face_landmarks=results # Store full results for drawing
        )


class DeepLabCutDetector:
    """
    Detector using a custom trained DeepLabCut model.
    Requires 'dlclive' to be installed and 'DLC_MODEL_PATH' in config.
    """
    def __init__(self, config):
        try:
            import dlclive
            from dlclive import DLCLive, Processor
        except ImportError:
            raise ImportError("DeepLabCut Live is not installed. Please run: pip install dlclive")

        self.config = config
        model_path = getattr(config, "DLC_MODEL_PATH", None)
        if not model_path:
            raise ValueError("DLC_MODEL_PATH not set in config.yaml")

        # Sanitize model_path in case user copied python raw string syntax (r"...") into YAML
        if isinstance(model_path, str):
            model_path = model_path.strip()
            # Remove r"..." or r'...' wrappers
            if (model_path.startswith('r"') and model_path.endswith('"')) or \
               (model_path.startswith("r'") and model_path.endswith("'")):
                model_path = model_path[2:-1]
            # Remove "..." or '...' wrappers if present without r
            elif (model_path.startswith('"') and model_path.endswith('"')) or \
                 (model_path.startswith("'") and model_path.endswith("'")):
                model_path = model_path[1:-1]

        # Load the label mapping from config.yml
        self.mapping = getattr(config, "DLC_LABEL_MAPPING", {})

        try:
            if getattr(config, "PRINT_DATA", True):
                print(f"DEBUG: Initializing DLCLive with model path: '{model_path}'")

            # The simple, robust way that matches the working demo script.
            # DLCLive is smart enough to determine the model type from the directory contents.
            # Forcing a model_type can cause issues if the path is a directory.
            self.dlc_proc = DLCLive(model_path)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize DLCLive with path: {model_path}\nError: {e}")

        # Extract body part names from the loaded configuration within DLCLive
        # Newer DLCLive versions loading .pt files store config internally
        if hasattr(self.dlc_proc, 'pose_cfg') and self.dlc_proc.pose_cfg:
            self.body_parts = self.dlc_proc.pose_cfg.get('all_joints_names', [])
        else:
            # Fallback: try to find pose_cfg.yaml manually if DLCLive didn't expose it
            search_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
            pose_cfg_path = os.path.join(search_dir, "pose_cfg.yaml")
            if os.path.exists(pose_cfg_path):
                with open(pose_cfg_path, 'r') as f:
                    self.body_parts = yaml.safe_load(f).get('all_joints_names', [])
            else:
                raise ValueError(f"Could not determine body parts. DLCLive object has no 'pose_cfg' and 'pose_cfg.yaml' not found in {search_dir}")

        # Initialize inference (warmup) with a dummy image
        # DLC expects height, width, channels
        self.dlc_proc.init_inference(np.zeros((int(config.cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
                                               int(config.cap.get(cv.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8))

    def detect(self, frame) -> Optional[FaceLandmarks]:
        # DLC Live returns a pose object. The structure depends on your export settings.
        # Usually it is a numpy array of shape (N, 3) -> [x, y, likelihood]
        pose = self.dlc_proc.get_pose(frame)

        landmarks_data = {}
        required_keys = [
            'left_iris_center', 'right_iris_center',
            'left_eye_outer_corner', 'right_eye_outer_corner',
            'nose_tip', 'chin',
            'left_mouth_corner', 'right_mouth_corner'
        ]

        for key in required_keys:
            dlc_label = self.mapping.get(key)

            # If label is not mapped or not found in model, we cannot form a complete face
            if not dlc_label or dlc_label not in self.body_parts:
                # You might want to log a warning here once
                return None

            idx = self.body_parts.index(dlc_label)

            # Check confidence
            confidence = pose[idx][2]
            if confidence >= self.config.MIN_DETECTION_CONFIDENCE:
                landmarks_data[key] = pose[idx][:2] # Store [x, y]

        # If we have no data at all, return None
        if not landmarks_data:
            return None
            
        return FaceLandmarks(**landmarks_data)

class Ocapi(object):
	def __init__(self, subject_id=None, config_file_path="config.yaml", VIDEO_INPUT=None, VIDEO_OUTPUT=None, WEBCAM=0,
	             TRACKING_DATA_LOG_FOLDER=None, starting_timestamp=None, total_frames=None,
	             eeg_trial_onsets_ms=None, session=None):

		# 2. Set up core attributes from arguments
		self.subject_id = subject_id
		self.session = session
		self.VIDEO_INPUT = VIDEO_INPUT
		self.VIDEO_OUTPUT_BASE = VIDEO_OUTPUT
		self.TRACKING_DATA_LOG_FOLDER = TRACKING_DATA_LOG_FOLDER
		self.WEBCAM = WEBCAM
		self.total_frames = total_frames or 0

		# Setup logging before any print statements
		self._setup_logging()

		# 1. Load all parameters from the YAML file into class attributes
		self.load_config(file_path=config_file_path)

		# 3. Set default values for any attributes that might be missing from the config
		# This makes the class more robust.
		# System & Performance
		self.PRINT_DATA = getattr(self, "PRINT_DATA", True)
		self.SHOW_ON_SCREEN_DATA = getattr(self, "SHOW_ON_SCREEN_DATA", True)
		self.TUNING_MODE = getattr(self, "TUNING_MODE", False)
		self.FRAME_SKIP = getattr(self, "FRAME_SKIP", 1)
		self.USE_SOCKET = getattr(self, "USE_SOCKET", False)
		self.SERVER_IP = getattr(self, "SERVER_IP", "127.0.0.1")
		self.SERVER_PORT = getattr(self, "SERVER_PORT", 5005)

		# Files & Logging
		self.ROTATE = getattr(self, "ROTATE", 0)
		self.FLIP_VIDEO = getattr(self, "FLIP_VIDEO", False)
		self.split_at_ms = getattr(self, "SPLIT_VIDEO_AT_MS", None)
		self.output_suffix_part1 = getattr(self, "OUTPUT_FILENAME_SUFFIX_PART1", "_part1")
		self.output_suffix_part2 = getattr(self, "OUTPUT_FILENAME_SUFFIX_PART2", "_part2")
		self.LOG_DATA = getattr(self, "LOG_DATA", True)
		self.TIMESTAMP_FORMAT = getattr(self, "TIMESTAMP_FORMAT", "%Y%m%d%H%M%S%f")
		self.LOG_ALL_FEATURES = getattr(self, "LOG_ALL_FEATURES", False)
		self.LOG_Z_COORD = getattr(self, "LOG_Z_COORD", False)

		# Features
		self.MAX_NUM_FACES = getattr(self, "MAX_NUM_FACES", 1)
		self.MIN_DETECTION_CONFIDENCE = getattr(self, "MIN_DETECTION_CONFIDENCE", 0.5)
		self.MIN_TRACKING_CONFIDENCE = getattr(self, "MIN_TRACKING_CONFIDENCE", 0.5)
		self.USE_ATTENTION_MESH = getattr(self, "USE_ATTENTION_MESH", True)
		self.ENABLE_HEAD_POSE = getattr(self, "ENABLE_HEAD_POSE", True)
		self.USER_FACE_WIDTH = getattr(self, "USER_FACE_WIDTH", 150.0)
		self.MOVING_AVERAGE_WINDOW = getattr(self, "MOVING_AVERAGE_WINDOW", 10)
		self.BLINK_THRESHOLD = getattr(self, "BLINK_THRESHOLD", 0.51)
		self.EYE_AR_CONSEC_FRAMES = getattr(self, "EYE_AR_CONSEC_FRAMES", 1)
		self.SHOW_ALL_FEATURES = getattr(self, "SHOW_ALL_FEATURES", False)

		# Landmark Indices
		self.LEFT_EYE_IRIS = getattr(self, "LEFT_EYE_IRIS", [474, 475, 476, 477])
		self.RIGHT_EYE_IRIS = getattr(self, "RIGHT_EYE_IRIS", [469, 470, 471, 472])
		self.RIGHT_EYE_POINTS = getattr(self, "RIGHT_EYE_POINTS", [33, 160, 159, 158, 133, 153, 145, 144])
		self.LEFT_EYE_POINTS = getattr(self, "LEFT_EYE_POINTS", [362, 385, 386, 387, 263, 373, 374, 380])
		self.RIGHT_EYE_OUTER_CORNER = getattr(self, "RIGHT_EYE_OUTER_CORNER", 33)
		self.LEFT_EYE_OUTER_CORNER = getattr(self, "LEFT_EYE_OUTER_CORNER", 263)
		self.NOSE_TIP_INDEX = getattr(self, "NOSE_TIP_INDEX", 4)
		self.CHIN_INDEX = getattr(self, "CHIN_INDEX", 152)
		self.RIGHT_MOUTH_CORNER = getattr(self, "RIGHT_MOUTH_CORNER", 61)
		self.LEFT_MOUTH_CORNER = getattr(self, "LEFT_MOUTH_CORNER", 291)
		self._indices_pose = getattr(self, "_indices_pose", [4, 152, 263, 33, 291, 61])

		# 4. Initialize hardware and models
		self.cap = self.init_video_input()
		self.FPS = self.cap.get(cv.CAP_PROP_FPS)
		if self.FPS == 0:
			self.logger.warning("Video FPS reported as 0. Defaulting to 30 FPS for calculations.")
			self.FPS = 30.0
		
		detector_type = getattr(self, "DETECTOR_TYPE", "mediapipe")
		if detector_type.lower() == "deeplabcut":
			self.detector = DeepLabCutDetector(self)
		else:
			self.detector = MediaPipeDetector(self)
		self.socket = self.init_socket()

		# 5. Initialize state variables
		self.initial_pitch, self.initial_yaw, self.initial_roll = None, None, None
		self.calibrated = False
		self.TOTAL_BLINKS = 0
		self.EYES_BLINK_FRAME_COUNTER = 0
		self.csv_data = []
		self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)
		self.current_video_part = 1
		self.split_triggered_and_finalized = False
		self._reset_per_frame_state()
		self._setup_column_names()

		# 6. Setup Calibration
		self.CALIBRATION_METHOD = getattr(self, "METHOD", "manual")
		self.auto_calibrate_pending = self.CALIBRATION_METHOD in ["clustering", "gaze_informed"]
		self.calib_duration_sec = getattr(self, "CLUSTERING_CALIB_DURATION_SECONDS", 30)
		self.CLUSTERING_CALIB_DURATION_FRAMES = int(self.calib_duration_sec * self.FPS)
		self.clustering_calib_all_samples = []
		self.head_pose_calibration_samples = {'pitch': [], 'yaw': [], 'roll': []}
		self.head_pose_calibration_frame_counter = 0

		# 7. Setup Trial Detection
		self.ENABLE_VIDEO_TRIAL_DETECTION = getattr(self, "ENABLE", False)
		if self.ENABLE_VIDEO_TRIAL_DETECTION:
			self.trial_counter = 0
			self.current_trial_data = None
			self.all_trials_summary = []
			self.last_trial_end_time_ms = -getattr(self, "MIN_INTER_TRIAL_INTERVAL_MS", 1000)
			self.roi_brightness_samples = []
			self.roi_baseline_mean = None
			self.roi_baseline_std_dev = None
			self.last_trial_result_text = ""
			# Load the new gaze threshold parameter with a safe default
			self.GAZE_DX_SUM_THRESHOLD = getattr(self, "GAZE_DX_SUM_THRESHOLD", 999)
			self._validate_trial_detection_config()

		# 8. Initialize video output for the first part
		current_output_suffix = self.output_suffix_part1 if self.split_at_ms is not None else ""
		if self.VIDEO_OUTPUT_BASE:
			self.out = self.init_video_output(part_suffix=current_output_suffix)
		else:
			self.out = None

		# 9. Set starting timestamp
		if not starting_timestamp:
			self.starting_timestamp = datetime.now()
		else:
			self.starting_timestamp = datetime.strptime(str(starting_timestamp), self.TIMESTAMP_FORMAT)

		# 10. Set up externally provided trial onsets
		self._eeg_trial_onsets_ms = None # Internal storage
		self.next_eeg_trial_index = 0
		if eeg_trial_onsets_ms:
			self.eeg_trial_onsets_ms = eeg_trial_onsets_ms # Use the property setter

	@property
	def eeg_trial_onsets_ms(self):
		"""Getter for the EEG trial onsets."""
		return self._eeg_trial_onsets_ms

	@eeg_trial_onsets_ms.setter
	def eeg_trial_onsets_ms(self, value):
		"""Setter for EEG trial onsets that also initializes the trial index."""
		self._eeg_trial_onsets_ms = value
		if self._eeg_trial_onsets_ms:
			self.next_eeg_trial_index = 0 # Reset the index whenever new onsets are provided
			if self.PRINT_DATA:
				self.logger.info(f"Set {len(self._eeg_trial_onsets_ms)} trial onsets from EEG data. Video trial detection will be bypassed.")

	def _setup_logging(self):
		"""Configures the logger for the tracker."""
		# Use a unique name for the logger to avoid conflicts.
		logger_name = f"Ocapi_{self.subject_id or 'default'}_{id(self)}"
		self.logger = logging.getLogger(logger_name)
		self.logger.setLevel(logging.INFO)  # Set the lowest level to capture all messages.

		# Prevent adding multiple handlers if the method is called again.
		if self.logger.hasHandlers():
			self.logger.handlers.clear()

		# --- File Handler ---
		log_folder = self.TRACKING_DATA_LOG_FOLDER or "."
		os.makedirs(log_folder, exist_ok=True)

		ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
		subj = self.subject_id or 'NA'
		log_filename = os.path.join(log_folder, f"{subj}_{self.session}_head_gaze_tracker_{ts_str}.log")

		file_handler = logging.FileHandler(log_filename)
		file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(file_formatter)
		self.logger.addHandler(file_handler)

		# --- Console Handler ---
		# Use the PRINT_DATA flag to control console output.
		if getattr(self, 'PRINT_DATA', True):
			console_handler = logging.StreamHandler()
			console_handler.setFormatter(logging.Formatter('%(message)s'))  # Simple format for console
			self.logger.addHandler(console_handler)
	
	def _reset_per_frame_state(self):
		"""Resets variables that store state for the current frame."""
		self.adj_pitch, self.adj_yaw, self.adj_roll = 0.0, 0.0, 0.0
		self.smooth_pitch, self.smooth_yaw, self.smooth_roll = 0.0, 0.0, 0.0
		self.raw_head_pitch, self.raw_head_yaw, self.raw_head_roll = 0.0, 0.0, 0.0

		self.l_cx, self.l_cy, self.r_cx, self.r_cy = 0, 0, 0, 0
		self.l_dx, self.l_dy, self.r_dx, self.r_dy = 0.0, 0.0, 0.0, 0.0

		self.face_looks_display_text = ""
		self.gaze_on_stimulus_display_text = ""
		self.is_looking_down_explicitly = False
		self.current_roi_brightness = 0.0

	def _setup_column_names(self):
		self.column_names = [
			"Timestamp (ms)", "Frame Nr",
			"Left Eye Center X", "Left Eye Center Y",
			"Right Eye Center X", "Right Eye Center Y",
			"Left Iris Relative Pos Dx", "Left Iris Relative Pos Dy",
			"Right Iris Relative Pos Dx", "Right Iris Relative Pos Dy",
			"Total Blink Count",
		]
		if self.ENABLE_HEAD_POSE:
			self.column_names.extend(["Pitch", "Yaw", "Roll"])
		if self.LOG_ALL_FEATURES:
			num_landmarks = 478 if self.USE_ATTENTION_MESH else 468
			self.column_names.extend(
				[f"Landmark_{i}_X" for i in range(num_landmarks)]
				+ [f"Landmark_{i}_Y" for i in range(num_landmarks)]
				+ ([f"Landmark_{i}_Z" for i in range(num_landmarks)] if self.LOG_Z_COORD else [])
			)

	def _validate_trial_detection_config(self):
		"""
		Validates that all necessary parameters for trial detection and gaze classification
		are present and correctly formatted in the loaded configuration.
		Raises a ValueError if a critical parameter is missing or malformed.
		"""
		# Helper function to reduce code repetition.
		def _check_param_is_list_of_two(param_name):
			if not hasattr(self, param_name) or not (
					isinstance(getattr(self, param_name), list) and len(getattr(self, param_name)) == 2):
				raise ValueError(
					f"Configuration Error: The parameter '{param_name}' is required for the selected gaze "
					f"classification method and must be a list of two numbers (e.g., [min, max])."
				)

		# --- 1. Validate Trial Onset Detection ---
		if not hasattr(self, 'ROI_BRIGHTNESS_THRESHOLD_FACTOR') and \
		   not hasattr(self, 'ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD'):
			self.logger.warning("Neither 'ROI_BRIGHTNESS_THRESHOLD_FACTOR' nor 'ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD' is set.")
			self.logger.warning("Trial detection via screen brightness might not work.")
			# Set a high default to prevent accidental triggering.
			self.ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD = 255

		roi_coords = getattr(self, 'STIMULUS_ROI_COORDS', None)
		if not (isinstance(roi_coords, list) and len(roi_coords) == 4):
			raise ValueError("Configuration Error: 'STIMULUS_ROI_COORDS' must be a list of 4 integers [x, y, w, h].")

		# --- 2. Validate Gaze Classification Parameters ---
		gaze_method = getattr(self, 'GAZE_CLASSIFICATION_METHOD', 'none')

		if gaze_method == "compensatory_gaze":
			required_params = [
				'COMPENSATORY_HEAD_PITCH_RANGE',
				'COMPENSATORY_HEAD_YAW_RANGE_CENTER', 'COMPENSATORY_EYE_SUM_RANGE_CENTER',
				'COMPENSATORY_HEAD_YAW_RANGE_LEFT', 'COMPENSATORY_EYE_SUM_RANGE_LEFT_TURN',
				'COMPENSATORY_HEAD_YAW_RANGE_RIGHT', 'COMPENSATORY_EYE_SUM_RANGE_RIGHT_TURN',
				'STIMULUS_LEFT_IRIS_DY_RANGE', 'STIMULUS_RIGHT_IRIS_DY_RANGE' # Still needed for vertical check
			]
			for param in required_params:
				_check_param_is_list_of_two(param)

		elif "eye_gaze" in gaze_method:
			# This block handles "eye_gaze_only" and "eye_gaze_with_head_filter"

			# Check for horizontal eye gaze params (either the new sum or the old ranges)
			if not hasattr(self, 'GAZE_DX_SUM_THRESHOLD') or self.GAZE_DX_SUM_THRESHOLD >= 999:
				# If sum threshold is disabled, the old individual ranges are required.
				_check_param_is_list_of_two('STIMULUS_LEFT_IRIS_DX_RANGE')
				_check_param_is_list_of_two('STIMULUS_RIGHT_IRIS_DX_RANGE')

			# Check for vertical eye gaze params (always required for eye_gaze methods)
			_check_param_is_list_of_two('STIMULUS_LEFT_IRIS_DY_RANGE')
			_check_param_is_list_of_two('STIMULUS_RIGHT_IRIS_DY_RANGE')

			# If using the head filter, check for its specific parameters.
			if gaze_method == "eye_gaze_with_head_filter":
				_check_param_is_list_of_two('HEAD_POSE_FILTER_PITCH_RANGE')
				_check_param_is_list_of_two('HEAD_POSE_FILTER_YAW_RANGE')

		elif gaze_method == "head_pose_only":
			_check_param_is_list_of_two('STIMULUS_PITCH_RANGE')
			_check_param_is_list_of_two('STIMULUS_YAW_RANGE')

		# --- 3. Validate Gaze Refinements (Non-critical, provide warnings) ---
		if not hasattr(self, 'DOWNWARD_LOOK_LEFT_IRIS_DY_MIN'):
			self.logger.warning("'DOWNWARD_LOOK_LEFT_IRIS_DY_MIN' not in config. Defaulting to 999 (disabled).")
			self.DOWNWARD_LOOK_LEFT_IRIS_DY_MIN = 999
		if not hasattr(self, 'DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN'):
			self.logger.warning("'DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN' not in config. Defaulting to 999 (disabled).")
			self.DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN = 999

	def load_config(self, file_path):
		"""Loads configuration from a nested YAML file and sets them as class attributes without prefixes."""
		try:
			with open(file_path, 'r') as file:
				config_data = yaml.safe_load(file)
			if not config_data:
				raise ValueError("Config file is empty or invalid.")

			# Iterate through the top-level sections (e.g., 'System', 'Files')
			for section, params in config_data.items():
				if isinstance(params, dict):
					# Iterate through the key-value pairs in each section
					for key, value in params.items():
						# If a value is another dictionary (like 'Clustering'), iterate through it too
						if isinstance(value, dict):
							for sub_key, sub_value in value.items():
								setattr(self, sub_key, sub_value)
						else:
							# Set the attribute directly on the class instance
							setattr(self, key, value)
				else:
					# For any top-level non-dictionary items
					setattr(self, section, params)

		except FileNotFoundError:
			self.logger.error(f"Configuration file not found at '{file_path}'")
			raise
		except (yaml.YAMLError, ValueError) as e:
			self.logger.error(f"Error parsing YAML configuration file: {e}")
			raise

	@staticmethod
	def get_eeg_stimulus_times(header_file, marker_file, stimulus_description, events_of_interest=None):
		"""
		Parses BrainVision header and marker files to find the absolute timestamp
		of the first stimulus event and all raw stimulus sample points.

		Args:
			header_file (str or pathlib.Path): Path to the .vhdr file.
			marker_file (str or pathlib.Path): Path to the .vmrk file.
			stimulus_description (str): The description of the stimulus marker (e.g., "Stimulus").
			events_of_interest (list of str, optional): A list of specific marker descriptions
				(e.g., ["S  1", "S  2"]) to extract. If None, all markers matching
				`stimulus_description` are returned. Defaults to None.

		Returns:
			tuple: (sampling_rate_hz, all_stim_samples)
				   - The EEG sampling rate in Hz.
				   - A list of raw, unmodified integer sample points for ALL stimuli.
		"""
		# --- 1. Read Sampling Rate from Header File (.vhdr) ---
		sampling_interval_us = None
		# Use 'utf-8-sig' to handle potential Byte Order Mark (BOM) at the start of the file
		with open(header_file, 'r', encoding='utf-8-sig') as f:
			for line in f:
				# Use a more specific regex to match ONLY the SamplingInterval line
				match = re.search(r'^SamplingInterval=(\d+)', line.strip())
				if match:
					sampling_interval_us = float(match.group(1))
					break
		if sampling_interval_us is None:
			raise ValueError("Could not find 'SamplingInterval' in the header file.")
		sampling_rate_hz = 1_000_000.0 / sampling_interval_us
		if events_of_interest:
			logging.info(f"Searching for specific EEG events: {events_of_interest}")
		logging.info(f"Found EEG Sampling Rate: {sampling_rate_hz:.2f} Hz...")

		# --- 2. Read Raw Marker Samples from .vmrk File ---
		all_stim_samples = []

		# (!!!) FIX: Normalize the events_of_interest list by removing all spaces.
		# This makes matching robust against formatting like "S  1", "S 1", or "S1".
		normalized_events_of_interest = {e.replace(" ", "") for e in events_of_interest} if events_of_interest else None

		with open(marker_file, 'r', encoding='utf-8-sig') as f:
			for line in f:
				# (!!!) FIX: Use a more robust regular expression to find stimulus lines.
				# This correctly handles variations like "Stimulus,S 22" and "Stimulus, S 22".
				# It looks for a line starting with 'Mk' followed by the description and a comma.
				if line.startswith('Mk') and re.search(f"={stimulus_description},", line):
					parts = line.strip().split(',') # e.g., ['Mk4=Stimulus', 'S 22', '7115', '1', '0']
					if len(parts) < 3: continue # Skip malformed lines

					# Normalize the marker from the file by removing all spaces.
					marker_content_normalized = parts[1].replace(" ", "")

					# If a filter list is provided, check if this marker is in it.
					# Otherwise, accept all markers of the specified type.
					if not normalized_events_of_interest or marker_content_normalized in normalized_events_of_interest:
						stim_sample = int(parts[2])
						all_stim_samples.append(stim_sample)

		if not all_stim_samples:
			error_msg = f"Could not find any '{stimulus_description}' markers"
			if events_of_interest:
				error_msg += f" matching the specified events_of_interest"
			raise ValueError(f"{error_msg} in the EEG log file.")

		logging.info(f"Found {len(all_stim_samples)} stimulus markers. First is at sample: {all_stim_samples[0]}")
		return sampling_rate_hz, all_stim_samples

	def sync_with_eeg_and_set_onsets(self, header_file, marker_file, stimulus_description, events_of_interest=None):
		"""
		Orchestrates the synchronization between video and EEG data to determine
		the precise, video-aligned timestamps for trial onsets.

		This method performs several steps:
		1. A fast pass over the video to find the first visual stimulus onset.
		2. Parses the EEG files to get the raw stimulus marker samples.
		3. Calculates the time offset between the video and EEG recordings.
		4. Adjusts all EEG marker times using this offset.
		5. Stores the final, synchronized trial onsets (in milliseconds) in the
		   `eeg_trial_onsets_ms` attribute, ready for the main `run()` method.

		Args:
			header_file (str or pathlib.Path): Path to the EEG .vhdr file.
			marker_file (str or pathlib.Path): Path to the EEG .vmrk file.
			stimulus_description (str): The description of the stimulus marker in the EEG file.
			events_of_interest (list of str, optional): A specific list of marker
				descriptions to use for trial onsets. If None, all stimuli are used.
				Defaults to None.

		Raises:
			RuntimeError: If the first stimulus cannot be found in the video.
		"""
		if self.PRINT_DATA:
			self.logger.info("--- Starting EEG Synchronization Process ---")

		# 1. Find the first stimulus onset time in the video
		_, video_stim_ms = self.find_first_stimulus_onset()
		if video_stim_ms is None:
			raise RuntimeError("Could not find the first stimulus in the video. Check ROI settings in config.")

		# 2. Get EEG data and calculate the offset
		eeg_sampling_rate, raw_eeg_samples = self.get_eeg_stimulus_times(
			header_file, marker_file, stimulus_description, events_of_interest)
		video_stim_samples = (video_stim_ms / 1000.0) * eeg_sampling_rate
		sample_offset = raw_eeg_samples[0] - video_stim_samples
		adjusted_eeg_samples = [s - sample_offset for s in raw_eeg_samples]
		final_eeg_onsets_ms = [int((s / eeg_sampling_rate) * 1000) for s in adjusted_eeg_samples]

		# 3. Set the calculated onsets on the instance and save sync info
		self.eeg_trial_onsets_ms = final_eeg_onsets_ms

		# --- NEW: Save synchronization info to a file for external analysis ---
		sync_info = {
			'eeg_sampling_rate': eeg_sampling_rate,
			'sample_offset': sample_offset,
			'video_stim_ms': video_stim_ms,
			'first_eeg_stim_sample_raw': raw_eeg_samples[0]
		}
		folder = self.TRACKING_DATA_LOG_FOLDER or "."
		subj = self.subject_id or 'NA'
		sync_info_filename = os.path.join(folder, f"{subj}_{self.session}_eeg_sync_info.json")
		os.makedirs(os.path.dirname(sync_info_filename), exist_ok=True)
		import json
		with open(sync_info_filename, 'w') as f:
			json.dump(sync_info, f, indent=4)
		self.logger.info(f"Saved EEG synchronization info to {sync_info_filename}")

	@staticmethod
	def vector_position(point1, point2):
		x1, y1 = point1.ravel()
		x2, y2 = point2.ravel()
		return x2 - x1, y2 - y1

	@staticmethod
	def euclidean_distance_3D(points):
		P0, P3, P4, P5, P8, P11, P12, P13 = points
		numerator = (np.linalg.norm(P3 - P13) ** 3 + np.linalg.norm(P4 - P12) ** 3 + np.linalg.norm(P5 - P11) ** 3)
		denominator = 3 * np.linalg.norm(P0 - P8) ** 3
		return numerator / denominator if denominator else 0

	def estimate_head_pose(self, landmarks: FaceLandmarks, image_size):
		scale_factor = self.USER_FACE_WIDTH / 150.0
		# This 3D model is a generic representation of a human head.
		# Map attribute names to 3D model points
		model_points_map = {
			'nose_tip': (0.0, 0.0, 0.0),
			'chin': (0.0, -330.0 * scale_factor, -65.0 * scale_factor),
			'left_eye_outer_corner': (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
			'right_eye_outer_corner': (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
			'left_mouth_corner': (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),
			'right_mouth_corner': (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)
		}

		focal_length = image_size[1]
		center = (image_size[1] / 2, image_size[0] / 2)
		camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
		                         dtype="double")
		dist_coeffs = np.zeros((4, 1))

		# Dynamically build the 2D-3D correspondences based on available landmarks
		image_points_list = []
		model_points_list = []

		for attr_name, point_3d in model_points_map.items():
			point_2d = getattr(landmarks, attr_name)
			if point_2d is not None:
				image_points_list.append(point_2d)
				model_points_list.append(point_3d)

		# We need at least 4 points for a reliable PnP solution
		if len(image_points_list) < 4:
			return 0.0, 0.0, 0.0

		image_points = np.array(image_points_list, dtype="double")
		model_points = np.array(model_points_list, dtype="double")

		try:
			(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
			                                                             dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
			if not success: return 0.0, 0.0, 0.0
			rotation_matrix, _ = cv.Rodrigues(rotation_vector)
			projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
			_, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
			pitch, yaw, roll = euler_angles.flatten()[:3]
			return self.normalize_pitch(pitch), yaw, roll
		except cv.error as e:
			return 0.0, 0.0, 0.0

	@staticmethod
	def normalize_pitch(pitch):
		if pitch > 180: pitch -= 360
		if pitch < -90:
			pitch = -(180 + pitch)
		elif pitch > 90:
			pitch = 180 - pitch
		return -pitch

	def blinking_ratio(self, landmarks):
		right_eye_ratio = self.euclidean_distance_3D(landmarks[self.RIGHT_EYE_POINTS])
		left_eye_ratio = self.euclidean_distance_3D(landmarks[self.LEFT_EYE_POINTS])
		return (right_eye_ratio + left_eye_ratio + 1) / 2

	def init_video_input(self):
		if self.WEBCAM is None and self.VIDEO_INPUT:
			cap = cv.VideoCapture(self.VIDEO_INPUT)
			if not cap or not cap.isOpened():
				raise IOError(f"Cannot open video file: {self.VIDEO_INPUT}")
		elif self.VIDEO_INPUT is None:
			cap = cv.VideoCapture(self.WEBCAM)
			if not cap or not cap.isOpened():
				raise IOError(f"Cannot open webcam: {self.WEBCAM}")
		else:
			raise ValueError("Provide EITHER VIDEO_INPUT OR WEBCAM, not both or neither.")
		return cap

	def init_video_output(self, part_suffix=""):
		if not self.VIDEO_OUTPUT_BASE: return None  # Use VIDEO_OUTPUT_BASE

		base, ext = os.path.splitext(self.VIDEO_OUTPUT_BASE)
		# If a suffix is provided (like _part1), VIDEO_OUTPUT_BASE itself might already have a suffix from batch processor
		# We need to ensure the part_suffix is added correctly.
		# Let's assume VIDEO_OUTPUT_BASE is the *final* intended filename *without* part suffix.
		# The batch processor should pass a VIDEO_OUTPUT path that is the base for this specific run.
		# So, if batch processor passes "output/videoA_processed.mp4",
		# and part_suffix is "_part1", we want "output/videoA_processed_part1.mp4".

		# Let's adjust how VIDEO_OUTPUT_BASE is used.
		# The VIDEO_OUTPUT parameter passed to __init__ should be the base for this run.
		# So, self.VIDEO_OUTPUT_BASE should be what's passed to __init__.

		final_video_output_path = self.VIDEO_OUTPUT_BASE  # This is the path passed to __init__
		if part_suffix:
			name, extension = os.path.splitext(final_video_output_path)
			final_video_output_path = f"{name}{part_suffix}{extension}"

		output_dir = os.path.dirname(final_video_output_path)
		if output_dir: os.makedirs(output_dir, exist_ok=True)

		frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
		frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		output_fps = self.cap.get(cv.CAP_PROP_FPS) or self.FPS

		fourcc_str = getattr(self, 'OUTPUT_VIDEO_FOURCC', 'XVID').upper()
		if len(fourcc_str) != 4:
			self.logger.warning(f"OUTPUT_VIDEO_FOURCC '{fourcc_str}' invalid. Defaulting to 'XVID'.")
			fourcc_str = 'XVID'
		fourcc = cv.VideoWriter_fourcc(*fourcc_str)

		if self.PRINT_DATA: self.logger.info(
			f"Initializing video output: {final_video_output_path} with FOURCC: {fourcc_str}, FPS: {output_fps:.2f}")
		writer = cv.VideoWriter(final_video_output_path, fourcc, output_fps, (frame_width, frame_height))
		if not writer.isOpened():
			self.logger.error(f"Could not open video writer for {final_video_output_path} with FOURCC {fourcc_str}.")
			return None
		return writer

	@staticmethod
	def init_socket():
		return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

	def _find_dynamic_roi(self):
		"""
		Analyzes a specific interval of the video to find the region with the highest
		brightness variance, which is assumed to be the stimulus ROI. This is a pre-analysis pass.
		Returns True on success, False on failure.
		"""
		if not self.cap or not self.cap.isOpened():
			self.logger.error("Video capture not open for dynamic ROI detection.")
			return False

		# --- 1. Get parameters from config ---
		search_start_sec = getattr(self, "SEARCH_START_SECONDS", 0)
		search_duration_sec = getattr(self, "SEARCH_DURATION_SECONDS", 30)
		grid_cols, grid_rows = getattr(self, "GRID_DIVISIONS", [12, 10])
		frame_limit = int(search_duration_sec * self.FPS)

		if self.PRINT_DATA:
			self.logger.info(f"Dynamic ROI search will start at {search_start_sec}s and run for {search_duration_sec}s.")
			self.logger.info(f"Using a {grid_cols}x{grid_rows} grid.")

		# --- 2. Seek to the start of the search interval ---
		if search_start_sec > 0:
			start_time_ms = search_start_sec * 1000
			self.cap.set(cv.CAP_PROP_POS_MSEC, start_time_ms)
			if self.PRINT_DATA:
				current_pos_ms = self.cap.get(cv.CAP_PROP_POS_MSEC)
				self.logger.info(f"Seeking video to {start_time_ms}ms... Current position is now ~{current_pos_ms:.0f}ms.")

		# --- 3. Collect brightness data ---
		cell_brightness_history = np.zeros((grid_rows, grid_cols, frame_limit), dtype=np.float32)
		frame_num = 0
		last_frame_for_viz = None
		while self.cap.isOpened() and frame_num < frame_limit:
			frame, img_h, img_w, ret = self._get_and_preprocess_frame()
			if not ret:
				if self.PRINT_DATA: self.logger.info("Video ended before ROI search period finished.")
				break
			last_frame_for_viz = frame.copy()
			search_area_coords = getattr(self, "SEARCH_AREA", [0, 0, img_w, img_h])
			x_s, y_s, w_s, h_s = search_area_coords
			search_frame = frame[y_s:y_s + h_s, x_s:x_s + w_s]
			gray_frame = cv.cvtColor(search_frame, cv.COLOR_BGR2GRAY)
			area_h, area_w = gray_frame.shape
			cell_h, cell_w = area_h // grid_rows, area_w // grid_cols
			for r in range(grid_rows):
				for c in range(grid_cols):
					y1, y2 = r * cell_h, (r + 1) * cell_h
					x1, x2 = c * cell_w, (c + 1) * cell_w
					cell = gray_frame[y1:y2, x1:x2]
					if cell.size > 0:
						cell_brightness_history[r, c, frame_num] = np.mean(cell)
			frame_num += 1
			if self.PRINT_DATA and frame_num % int(self.FPS or 30) == 0:
				self.logger.info(f"  ROI Search Pass: Processed {frame_num}/{frame_limit} frames...")

		# --- 4. Analyze the collected data ---
		if frame_num == 0:
			self.logger.error("No frames processed for ROI detection. The search start time may be past the end of the video.")
			return False
		cell_brightness_history = cell_brightness_history[:, :, :frame_num]
		stds = np.std(cell_brightness_history, axis=2)
		max_std = np.max(stds)
		if max_std < 2.0:
			self.logger.warning("Dynamic ROI detection found very low activity in the search interval.")
			self.cap.release()
			self.cap = self.init_video_input()
			return True

		# --- 5. Create initial activity map ---
		activity_threshold_percent = getattr(self, "ACTIVITY_THRESHOLD_PERCENT", 40)
		activity_threshold = max_std * (activity_threshold_percent / 100.0)
		activity_map = (stds >= activity_threshold).astype(np.uint8)

		# --- 6. Clean the activity map ---
		# (!!!) FIX: Access cleaning parameters from the MorphologicalCleaning dictionary
		cleaning_config = getattr(self, "MorphologicalCleaning", {})
		if cleaning_config.get("ENABLE_CLEANING", False):
			# --- 6a. Morphological fine-tuning (breaks bridges to outliers) ---
			erode_iter = cleaning_config.get("ERODE_ITERATIONS", 1)
			dilate_iter = cleaning_config.get("DILATE_ITERATIONS", 2)
			if erode_iter > 0:
				kernel = np.ones((3, 3), np.uint8)
				activity_map = cv.erode(activity_map, kernel, iterations=erode_iter)
			if dilate_iter > 0:
				kernel = np.ones((3, 3), np.uint8)
				activity_map = cv.dilate(activity_map, kernel, iterations=dilate_iter)

			# --- 6b. Isolate the largest blob after cleaning ---
			if cleaning_config.get("KEEP_LARGEST_BLOB_ONLY", False):
				contours, _ = cv.findContours(activity_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
				if contours:
					largest_contour = max(contours, key=cv.contourArea)
					clean_activity_map = np.zeros_like(activity_map)
					cv.drawContours(clean_activity_map, [largest_contour], -1, 1, -1)
					activity_map = clean_activity_map
					if self.PRINT_DATA:
						self.logger.info(f"Isolated largest activity blob, removing {len(contours) - 1} smaller outlier(s).")

		# --- 7. Calculate bounding box from the cleaned map ---
		active_cells_indices = np.argwhere(activity_map > 0)
		if active_cells_indices.size == 0:
			self.logger.warning("No cells remained after cleaning. Falling back to the single most active cell.")
			active_cells_indices = np.array([np.unravel_index(np.argmax(stds), stds.shape)])

		min_row, max_row = np.min(active_cells_indices[:, 0]), np.max(active_cells_indices[:, 0])
		min_col, max_col = np.min(active_cells_indices[:, 1]), np.max(active_cells_indices[:, 1])

		x_s, y_s, w_s, h_s = getattr(self, "SEARCH_AREA", [0, 0, last_frame_for_viz.shape[1], last_frame_for_viz.shape[0]])
		search_frame_shape = (h_s, w_s)
		area_h, area_w = search_frame_shape
		cell_h, cell_w = area_h // grid_rows, area_w // grid_cols

		roi_x = (min_col * cell_w) + x_s
		roi_y = (min_row * cell_h) + y_s
		roi_w = (max_col - min_col + 1) * cell_w
		roi_h = (max_row - min_row + 1) * cell_h

		# --- 8. Apply Padding (Optional) ---
		# (!!!) FIX: Access padding parameters from the ROIPadding dictionary
		padding_config = getattr(self, "ROIPadding", {})
		if padding_config.get("ENABLE_PADDING", False):
			padding_percent = padding_config.get("PADDING_PERCENTAGE", 0)
			if padding_percent != 0: # Allow both positive (expand) and negative (shrink) padding
				orig_roi_x, orig_roi_y, orig_roi_w, orig_roi_h = roi_x, roi_y, roi_w, roi_h
				pad_w = int(roi_w * (padding_percent / 100.0))
				pad_h = int(roi_h * (padding_percent / 100.0))
				roi_x -= pad_w // 2
				roi_y -= pad_h // 2
				roi_w += pad_w
				roi_h += pad_h
				img_h_total, img_w_total, _ = last_frame_for_viz.shape
				roi_x = max(0, roi_x)
				roi_y = max(0, roi_y)
				if roi_x + roi_w > img_w_total: roi_w = img_w_total - roi_x
				if roi_y + roi_h > img_h_total: roi_h = img_h_total - roi_y
				if self.PRINT_DATA:
					action = "expanded" if padding_percent > 0 else "shrunk"
					self.logger.info(f"Applied {padding_percent}% padding. ROI {action}.")

		# --- 9. Finalize and Visualize ---
		self.STIMULUS_ROI_COORDS = [roi_x, roi_y, roi_w, roi_h]
		if getattr(self, "VISUALIZE_ROI_SEARCH", False) and last_frame_for_viz is not None:
			# (Visualization logic remains the same)
			viz_img = last_frame_for_viz.copy()
			overlay = viz_img.copy()
			cv.rectangle(viz_img, (x_s, y_s), (x_s + w_s, y_s + h_s), (255, 255, 0), 2)
			for r in range(grid_rows):
				for c in range(grid_cols):
					if activity_map[r, c] > 0:
						intensity = min(stds[r, c] / max_std, 1.0)
						color = (0, int(155 + intensity * 100), 0)
						cell_x, cell_y = (c * cell_w) + x_s, (r * cell_h) + y_s
						cv.rectangle(overlay, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), color, -1)
			cv.addWeighted(overlay, 0.5, viz_img, 0.5, 0, viz_img)
			if 'orig_roi_x' in locals():
				cv.rectangle(viz_img, (orig_roi_x, orig_roi_y), (orig_roi_x + orig_roi_w, orig_roi_y + orig_roi_h), (0, 255, 255), 2)
				cv.putText(viz_img, "Detected Box", (orig_roi_x + 5, orig_roi_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
			cv.rectangle(viz_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 3)
			cv.putText(viz_img, "Final ROI", (roi_x + 5, roi_y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
			cv.imshow("Dynamic ROI Search Visualization", viz_img)
			self.logger.info("\n--- ROI VISUALIZATION ---")
			self.logger.info("Showing visualization of the ROI search. Press any key in the window to continue...")
			cv.waitKey(0)
			cv.destroyWindow("Dynamic ROI Search Visualization")

		if self.PRINT_DATA:
			self.logger.info("-" * 30)
			self.logger.info(f"Dynamic ROI detection complete.")
			self.logger.info(f"Setting STIMULUS_ROI_COORDS to final bounding box: {self.STIMULUS_ROI_COORDS}")
			self.logger.info("-" * 30)

		# IMPORTANT: Reset the video capture for the next pass
		self.cap.release()
		self.cap = self.init_video_input()
		return True

	def _get_and_preprocess_frame(self):
		ret, frame = self.cap.read()
		if not ret:
			return None, 0, 0, False

		if self.FLIP_VIDEO: frame = cv.flip(frame, 1)
		if self.ROTATE == 90:
			frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
		elif self.ROTATE == 180:
			frame = cv.rotate(frame, cv.ROTATE_180)
		elif self.ROTATE == -90 or self.ROTATE == 270:
			frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

		img_h, img_w = frame.shape[:2]
		return frame, img_h, img_w, True

	def _extract_eye_features(self, landmarks: FaceLandmarks):
		try:
			# Use pre-calculated centers from the detector
			l_cx_f, l_cy_f = landmarks.left_iris_center
			r_cx_f, r_cy_f = landmarks.right_iris_center
			
			self.l_cx, self.l_cy = int(l_cx_f), int(l_cy_f)
			self.r_cx, self.r_cy = int(r_cx_f), int(r_cy_f)
			center_left = np.array([l_cx_f, l_cy_f], dtype=np.float32)
			center_right = np.array([r_cx_f, r_cy_f], dtype=np.float32)
			
			outer_left_corner = landmarks.left_eye_outer_corner
			outer_right_corner = landmarks.right_eye_outer_corner
			
			self.l_dx, self.l_dy = self.vector_position(outer_left_corner, center_left)
			self.r_dx, self.r_dy = self.vector_position(outer_right_corner, center_right)
		except Exception:
			self.l_cx, self.l_cy, self.r_cx, self.r_cy = 0, 0, 0, 0
			self.l_dx, self.l_dy, self.r_dx, self.r_dy = 0.0, 0.0, 0.0, 0.0

	def _check_downward_look(self):
		if hasattr(self, 'DOWNWARD_LOOK_LEFT_IRIS_DY_MIN') and hasattr(self, 'DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN'):
			if (self.l_dy > self.DOWNWARD_LOOK_LEFT_IRIS_DY_MIN and
					self.r_dy > self.DOWNWARD_LOOK_RIGHT_IRIS_DY_MIN):
				return True
		return False

	def _update_blink_count(self, mesh_points_3D_normalized):
		eyes_aspect_ratio = self.blinking_ratio(mesh_points_3D_normalized)
		if eyes_aspect_ratio <= self.BLINK_THRESHOLD:
			if not self.is_looking_down_explicitly:
				self.EYES_BLINK_FRAME_COUNTER += 1
			else:
				if self.EYES_BLINK_FRAME_COUNTER > 0:
					self.EYES_BLINK_FRAME_COUNTER = 0
		else:
			if self.EYES_BLINK_FRAME_COUNTER > self.EYE_AR_CONSEC_FRAMES:
				self.TOTAL_BLINKS += 1
			self.EYES_BLINK_FRAME_COUNTER = 0

	def _process_head_pose(self, landmarks: FaceLandmarks, img_h, img_w, key_pressed):
		# Estimate raw head pose and apply moving average for smoothing
		self.raw_head_pitch, self.raw_head_yaw, self.raw_head_roll = self.estimate_head_pose(landmarks,
		                                                                                     (img_h, img_w))
		self.angle_buffer.add([self.raw_head_pitch, self.raw_head_yaw, self.raw_head_roll])
		self.smooth_pitch, self.smooth_yaw, self.smooth_roll = self.angle_buffer.get_average()

		# --- Automatic Head Pose Calibration Logic ---
		# This block runs if auto-calibration is pending and a baseline hasn't been set yet.
		if self.auto_calibrate_pending and not self.calibrated:

			# --- Method 1: Clustering (most robust for noisy data) ---
			if self.CALIBRATION_METHOD == 'clustering':
				# Collect all smoothed head poses for the configured duration
				self.clustering_calib_all_samples.append([self.smooth_pitch, self.smooth_yaw, self.smooth_roll])

				# Once enough samples are collected, perform the clustering analysis
				if len(self.clustering_calib_all_samples) >= self.CLUSTERING_CALIB_DURATION_FRAMES:
					self._perform_clustering_calibration()
					self.auto_calibrate_pending = False  # Calibration attempt is complete

			# --- Method 2: Gaze-Informed (collects samples when eyes are centered) ---
			elif self.CALIBRATION_METHOD == 'gaze_informed':
				self.head_pose_calibration_frame_counter += 1

				# Check if eyes are looking relatively forward to ensure a good quality sample
				eye_dx_ok = (self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[0] <= self.l_dx <=
				             self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[1] and
				             self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[0] <= self.r_dx <=
				             self.HEAD_POSE_AUTO_CALIB_EYE_DX_RANGE[1])
				eye_dy_ok = (self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[0] <= self.l_dy <=
				             self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[1] and
				             self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[0] <= self.r_dy <=
				             self.HEAD_POSE_AUTO_CALIB_EYE_DY_RANGE[1])

				if eye_dx_ok and eye_dy_ok:
					self.head_pose_calibration_samples['pitch'].append(self.smooth_pitch)
					self.head_pose_calibration_samples['yaw'].append(self.smooth_yaw)
					self.head_pose_calibration_samples['roll'].append(self.smooth_roll)
					if self.PRINT_DATA and self.frame_count % 15 == 0:  # Occasional print
						self.logger.info(
							f"HP Calib sample: P={self.smooth_pitch:.1f} Y={self.smooth_yaw:.1f}. N={len(self.head_pose_calibration_samples['pitch'])}")

				# Check if the calibration period is over
				if self.head_pose_calibration_frame_counter >= self.HEAD_POSE_AUTO_CALIB_DURATION_FRAMES:
					# Ensure we have enough high-quality samples
					if len(self.head_pose_calibration_samples['pitch']) >= self.HEAD_POSE_AUTO_CALIB_MIN_SAMPLES:
						# Use MEDIAN instead of MEAN for robustness against outliers
						self.initial_pitch = np.mean(self.head_pose_calibration_samples['pitch'])
						self.initial_yaw = np.mean(self.head_pose_calibration_samples['yaw'])
						self.initial_roll = np.mean(self.head_pose_calibration_samples['roll'])
						self.calibrated = True
						if self.PRINT_DATA:
							self.logger.info(
								f"Head pose auto-calibrated using {len(self.head_pose_calibration_samples['pitch'])} samples.")
							self.logger.info(
								f"Initial Pose (Median): P={self.initial_pitch:.1f}, Y={self.initial_yaw:.1f}, R={self.initial_roll:.1f}")
					else:
						if self.PRINT_DATA:
							self.logger.warning(
								f"Head pose auto-calibration failed: Not enough suitable samples ({len(self.head_pose_calibration_samples['pitch'])} collected). Manual calibration ('c') may be needed.")

					self.auto_calibrate_pending = False  # Stop trying to auto-calibrate to prevent repeated messages

		# Manual calibration by key press (overrides any auto-calibration)
		if key_pressed == ord('c'):
			self.initial_pitch, self.initial_yaw, self.initial_roll = self.smooth_pitch, self.smooth_yaw, self.smooth_roll
			self.calibrated = True
			self.auto_calibrate_pending = False  # Manual calibration also means we stop pending auto-calib
			if self.PRINT_DATA: self.logger.info(
				f"Head pose recalibrated by user: P={self.initial_pitch:.1f}, Y={self.initial_yaw:.1f}, R={self.initial_roll:.1f}")

		# Adjust head pose angles based on the calibration baseline
		if self.calibrated:
			self.adj_pitch = self.smooth_pitch - self.initial_pitch
			self.adj_yaw = self.smooth_yaw - self.initial_yaw
			self.adj_roll = self.smooth_roll - self.initial_roll
		else:
			# If not calibrated, use the raw smoothed values (no adjustment)
			self.adj_pitch, self.adj_yaw, self.adj_roll = self.smooth_pitch, self.smooth_yaw, self.smooth_roll

	def _perform_clustering_calibration(self):
		"""Finds the baseline pose by clustering all collected samples. Returns the DBSCAN labels."""
		if not self.clustering_calib_all_samples:
			if self.PRINT_DATA: self.logger.warning("Clustering calibration failed: No samples collected.")
			return None

		pose_array = np.array(self.clustering_calib_all_samples)
		if self.PRINT_DATA:
			self.logger.info(f"Performing clustering calibration on {len(pose_array)} head pose samples...")

		# DBSCAN parameters can be loaded from config
		eps = getattr(self, "CLUSTERING_DBSCAN_EPS", 3.0)
		min_samples = getattr(self, "CLUSTERING_DBSCAN_MIN_SAMPLES", 15)

		db = DBSCAN(eps=eps, min_samples=min_samples).fit(pose_array)
		labels = db.labels_

		# Find the largest cluster (ignoring noise, which is labeled -1)
		unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

		if len(counts) == 0:
			if self.PRINT_DATA:
				self.logger.warning("Clustering found no stable pose. Falling back to median of all samples.")
			# Fallback: if no clusters are found, the median of all data is still a robust choice.
			baseline_pose = np.median(pose_array, axis=0)
		else:
			# The baseline pose is the centroid (median) of the largest cluster
			largest_cluster_label = unique_labels[counts.argmax()]
			largest_cluster_points = pose_array[labels == largest_cluster_label]
			baseline_pose = np.median(largest_cluster_points, axis=0)
			if self.PRINT_DATA:
				self.logger.info(f"Found largest cluster with {counts.max()} points (out of {len(labels)}).")

		self.initial_pitch, self.initial_yaw, self.initial_roll = baseline_pose
		self.calibrated = True
		if self.PRINT_DATA:
			self.logger.info(f"Calibration completed via clustering.")
			self.logger.info(f"New Initial Pose: P={self.initial_pitch:.1f}, Y={self.initial_yaw:.1f}, R={self.initial_roll:.1f}")

		return labels

	def find_first_stimulus_onset(self):
		"""
		Performs a fast pass over the video to find the frame number and timestamp
		of the first stimulus onset based on ROI brightness.

		Returns:
			A tuple (frame_number, timestamp_ms) if a stimulus is found.
			A tuple (None, None) if no stimulus is found or an error occurs.
		"""
		if not self.ENABLE_VIDEO_TRIAL_DETECTION:
			self.logger.error("`ENABLE_VIDEO_TRIAL_DETECTION` must be true in config to find stimulus onset.")
			return None, None

		# --- NEW: Perform Clustering Calibration if configured, just like the main run() method ---
		# This ensures that if the main analysis relies on a calibrated pose, the same
		# calibration is available and consistent from the very beginning.
		if self.CALIBRATION_METHOD == 'clustering':
			self.logger.info("--- Running Clustering Calibration as part of stimulus onset search ---")
			if not self._execute_calibration_pass():
				self.logger.error("Clustering calibration failed during stimulus search. Aborting.")
				self._cleanup(finalize_data=False)
				return None, None
			# CRITICAL FIX: The calibration pass releases the video capture.
			# We must re-initialize it before the next step.
			self.logger.info("Re-initializing video capture after calibration pass...")
			self.cap = self.init_video_input()

		# --- NEW: Perform Dynamic ROI search if configured, just like the main run() method ---
		# This ensures the ROI used for finding the first stimulus is the same one used for the full analysis.
		if getattr(self, 'STIMULUS_ROI_METHOD', 'static') == 'dynamic':
			self.logger.info("--- Running Dynamic ROI Detection for stimulus onset search ---")
			if not self._find_dynamic_roi():
				self.logger.error("Dynamic ROI detection failed during stimulus search. Aborting.")
				self._cleanup(finalize_data=False)
				return None, None

		self.logger.info("--- Starting Fast Pass: Searching for first stimulus onset ---")
		self._reset_analysis_state()  # Ensure a clean state

		frame_count = -1
		try:
			while self.cap.isOpened():
				frame_count += 1
				frame, img_h, img_w, ret = self._get_and_preprocess_frame()
				if not ret:
					self.logger.warning("Video ended before any stimulus was detected.")
					return None, None

				current_frame_time_ms = int(frame_count * (1000.0 / self.FPS))

				# --- Simplified Trial Detection Logic ---
				self.current_roi_brightness = self._calculate_roi_brightness(frame, img_h, img_w)

				# 1. Collect baseline
				if self.roi_baseline_mean is None:
					if len(self.roi_brightness_samples) < self.ROI_BRIGHTNESS_BASELINE_FRAMES:
						self.roi_brightness_samples.append(self.current_roi_brightness)
					else:
						self.roi_baseline_mean = np.mean(self.roi_brightness_samples)
						self.roi_baseline_std_dev = np.std(self.roi_brightness_samples)
						if self.roi_baseline_std_dev < 0.5: self.roi_baseline_std_dev = 0.5
						if self.PRINT_DATA:
							self.logger.info(f"ROI Baseline Calculated: Mean={self.roi_baseline_mean:.2f}, SD={self.roi_baseline_std_dev:.2f}")
					continue # Continue to next frame while collecting baseline

				# 2. Detect stimulus (copied from _update_trial_state)
				stimulus_detected = False
				method = getattr(self, "TRIAL_ONSET_DETECTION_METHOD", "factor")
				if method == "factor":
					threshold = self.roi_baseline_mean * getattr(self, "ROI_BRIGHTNESS_THRESHOLD_FACTOR", 1.5)
					if self.current_roi_brightness > threshold: stimulus_detected = True
				elif method == "absolute":
					threshold = getattr(self, "ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD", 255)
					if self.current_roi_brightness > threshold: stimulus_detected = True
				elif method == "statistical":
					# Get the statistical threshold (N * std_dev)
					std_dev_multiplier = getattr(self, "ROI_BRIGHTNESS_STD_DEV_THRESHOLD", 5.0)
					statistical_threshold = self.roi_baseline_mean + (std_dev_multiplier * self.roi_baseline_std_dev)

					# Get the minimum factor threshold as a safety net
					min_factor = getattr(self, "ROI_BRIGHTNESS_MIN_FACTOR", 1.0)  # Default to 1.0 (no effect)
					factor_threshold = self.roi_baseline_mean * min_factor

					# The final threshold is the greater of the two, ensuring robustness.
					final_threshold = max(statistical_threshold, factor_threshold)

					if self.current_roi_brightness > final_threshold:
						stimulus_detected = True

				# 3. If detected, we're done!
				if stimulus_detected:
					self.logger.info(f"--- First stimulus detected at frame {frame_count} ({current_frame_time_ms} ms) ---")
					return frame_count, current_frame_time_ms

		finally:
			# --- MODIFICATION ---
			# Instead of tearing down, reset the video to the beginning and clear
			# only the trial state, preserving the calibration and ROI results.
			self.logger.info("First stimulus found. Resetting video to frame 0 for main analysis pass.")
			self._reset_for_main_pass()

	def _get_face_looks_text(self):
		angle_y = self.adj_yaw if self.calibrated else self.smooth_yaw
		angle_x = self.adj_pitch if self.calibrated else self.smooth_pitch
		threshold = 10
		if angle_y < -threshold: return "Face: Left"
		if angle_y > threshold: return "Face: Right"
		if angle_x < -threshold: return "Face: Down"
		if angle_x > threshold: return "Face: Up"
		return "Face: Forward"

	def _calculate_roi_brightness(self, frame, img_h, img_w):
		x, y, w, h = self.STIMULUS_ROI_COORDS
		x, y = max(0, x), max(0, y)
		roi_x2, roi_y2 = min(img_w, x + w), min(img_h, y + h)
		if roi_x2 > x and roi_y2 > y:
			stimulus_roi = frame[y:roi_y2, x:roi_x2]
			gray_stimulus_roi = cv.cvtColor(stimulus_roi, cv.COLOR_BGR2GRAY)
			return np.mean(gray_stimulus_roi)
		if self.PRINT_DATA and self.frame_count % 100 == 0:
			self.logger.warning(f"STIMULUS_ROI_COORDS {self.STIMULUS_ROI_COORDS} invalid for frame size ({img_w}x{img_h}).")
		return 0.0

	def _update_trial_state(self, current_frame_time_ms):
		"""
		Manages the trial state machine: calculates the brightness baseline, detects trial onsets,
		and finalizes trials when they end.
		"""
		# --- 1. Baseline Calculation ---
		# If the baseline hasn't been calculated yet, collect brightness samples.
		if self.roi_baseline_mean is None:
			# Only collect samples if a trial is not currently active.
			if not (self.current_trial_data and self.current_trial_data['active']):
				if len(self.roi_brightness_samples) < self.ROI_BRIGHTNESS_BASELINE_FRAMES:
					self.roi_brightness_samples.append(self.current_roi_brightness)
				# Once enough samples are collected, calculate the baseline statistics.
				elif self.roi_brightness_samples:
					self.roi_baseline_mean = np.mean(self.roi_brightness_samples)
					self.roi_baseline_std_dev = np.std(self.roi_brightness_samples)
					# A very low SD (e.g., from a solid black screen) can be problematic.
					# Ensure it's at least a small value to prevent division by zero or hypersensitivity.
					if self.roi_baseline_std_dev < 0.5:
						self.roi_baseline_std_dev = 0.5
					if self.PRINT_DATA:
						self.logger.info(
							f"ROI Baseline Calculated: Mean={self.roi_baseline_mean:.2f}, SD={self.roi_baseline_std_dev:.2f}")
			else:
				# If we are in a trial before baseline is set, do nothing.
				pass

		# --- 2. Trial Onset Detection ---
		stimulus_detected = False
		# Only try to detect a stimulus if the baseline has been established.
		if self.roi_baseline_mean is not None:
			method = getattr(self, "TRIAL_ONSET_DETECTION_METHOD", "factor")

			if method == "statistical":
				# Get the statistical threshold (N * std_dev)
				std_dev_multiplier = getattr(self, "ROI_BRIGHTNESS_STD_DEV_THRESHOLD", 5.0)
				statistical_threshold = self.roi_baseline_mean + (std_dev_multiplier * self.roi_baseline_std_dev)

				# Get the minimum factor threshold as a safety net
				min_factor = getattr(self, "ROI_BRIGHTNESS_MIN_FACTOR", 1.0)  # Default to 1.0 (no effect)
				factor_threshold = self.roi_baseline_mean * min_factor

				# The final threshold is the greater of the two, ensuring robustness.
				final_threshold = max(statistical_threshold, factor_threshold)

				if self.current_roi_brightness > final_threshold:
					stimulus_detected = True

			elif method == "factor":
				factor = getattr(self, "ROI_BRIGHTNESS_THRESHOLD_FACTOR", 1.5)
				threshold = self.roi_baseline_mean * factor
				if self.current_roi_brightness > threshold:
					stimulus_detected = True
			elif method == "absolute":
				threshold = getattr(self, "ROI_ABSOLUTE_BRIGHTNESS_THRESHOLD", 255)
				if self.current_roi_brightness > threshold:
					stimulus_detected = True

		# --- 3. Manage Trial State Machine ---
		trial_can_start = (current_frame_time_ms >= self.last_trial_end_time_ms + self.MIN_INTER_TRIAL_INTERVAL_MS)

		# If a stimulus is detected and we can start a new trial...
		if trial_can_start and stimulus_detected and not (
				self.current_trial_data and self.current_trial_data['active']):
			self.trial_counter += 1
			start_t = current_frame_time_ms
			stim_end_t = start_t + self.STIMULUS_DURATION_MS
			trial_end_t = stim_end_t + self.POST_STIMULUS_TRIAL_DURATION_MS
			self.current_trial_data = {
				'id': self.trial_counter, 'start_time_ms': start_t,
				'stimulus_end_time_ms': stim_end_t, 'trial_end_time_ms': trial_end_t,
				'active': True, 'stimulus_frames_processed_gaze': 0,
				'frames_on_stimulus_area': 0, 'looked_final': 1}
			if self.PRINT_DATA:
				self.logger.info(
					f"Trial {self.trial_counter} START @{start_t}ms (Part {self.current_video_part}). ROI Bright: {self.current_roi_brightness:.2f}")

		# If a trial is currently active, check if it should end.
		if self.current_trial_data and self.current_trial_data['active']:
			if current_frame_time_ms >= self.current_trial_data['trial_end_time_ms']:
				if self.PRINT_DATA:
					self.logger.info(
						f"Trial {self.current_trial_data['id']} END @{current_frame_time_ms}ms (Part {self.current_video_part}).")

				# Finalize trial classification (looked vs. away)
				if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
					perc_on_stim = (self.current_trial_data['frames_on_stimulus_area'] /
					                self.current_trial_data['stimulus_frames_processed_gaze']) * 100
					# If the percentage is below the threshold, classify as 'away' (2).
					# Otherwise, it remains the default 'looked' (1).
					if perc_on_stim < self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
						self.current_trial_data['looked_final'] = 2

				trial_id = self.current_trial_data['id']
				self.last_trial_result_text = f"Trial {trial_id} Result: {'Looked (1)' if self.current_trial_data['looked_final'] == 1 else 'Away (2)'}"

				self.all_trials_summary.append(self.current_trial_data.copy())
				self.last_trial_end_time_ms = self.current_trial_data['trial_end_time_ms']
				self.current_trial_data = None

	def _end_active_trial_if_needed(self, current_frame_time_ms):
		"""
		Checks if an active trial has passed its end time and finalizes it.
		This logic is now separate so it can be called for both EEG and video-triggered trials.
		"""
		if self.current_trial_data and self.current_trial_data['active']:
			if current_frame_time_ms >= self.current_trial_data['trial_end_time_ms']:
				if self.PRINT_DATA:
					self.logger.info(
						f"Trial {self.current_trial_data['id']} END @{current_frame_time_ms}ms (Part {self.current_video_part}).")

				# Finalize trial classification (looked vs. away)
				if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
					perc_on_stim = (self.current_trial_data['frames_on_stimulus_area'] /
					                self.current_trial_data['stimulus_frames_processed_gaze']) * 100
					# If the percentage is below the threshold, classify as 'away' (2).
					# Otherwise, it remains the default 'looked' (1).
					if perc_on_stim < self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
						self.current_trial_data['looked_final'] = 2

				trial_id = self.current_trial_data['id']
				self.last_trial_result_text = f"Trial {trial_id} Result: {'Looked (1)' if self.current_trial_data['looked_final'] == 1 else 'Away (2)'}"

				self.all_trials_summary.append(self.current_trial_data.copy())
				self.last_trial_end_time_ms = self.current_trial_data['trial_end_time_ms']
				self.current_trial_data = None

	def _check_for_eeg_trial_onset(self, current_frame_time_ms):
		"""
		Checks if the current frame time triggers a pre-defined EEG trial onset.
		This bypasses video-based detection.
		"""
		if not self.eeg_trial_onsets_ms or self.next_eeg_trial_index >= len(self.eeg_trial_onsets_ms):
			return # No more EEG trials to process

		# Get the timestamp of the next scheduled trial
		next_trial_start_ms = self.eeg_trial_onsets_ms[self.next_eeg_trial_index]

		# Check if a trial is already active or if it's too soon to start another
		trial_can_start = (current_frame_time_ms >= self.last_trial_end_time_ms + self.MIN_INTER_TRIAL_INTERVAL_MS)
		is_new_trial = not (self.current_trial_data and self.current_trial_data['active'])

		# If the current frame time has reached or passed the next trial's start time
		if trial_can_start and is_new_trial and current_frame_time_ms >= next_trial_start_ms:
			self.trial_counter += 1
			start_t = next_trial_start_ms # Use the precise EEG time
			stim_end_t = start_t + self.STIMULUS_DURATION_MS
			trial_end_t = stim_end_t + self.POST_STIMULUS_TRIAL_DURATION_MS
			self.current_trial_data = {
				'id': self.trial_counter, 'start_time_ms': start_t,
				'stimulus_end_time_ms': stim_end_t, 'trial_end_time_ms': trial_end_t,
				'active': True, 'stimulus_frames_processed_gaze': 0, 'frames_on_stimulus_area': 0,
				'looked_final': 1}
			self.next_eeg_trial_index += 1 # Move to the next trial in the list
			if self.PRINT_DATA:
				self.logger.info(f"EEG Trial {self.trial_counter} START @{start_t}ms (Part {self.current_video_part}).")

	def _classify_gaze_for_current_trial(self, current_frame_time_ms):
		"""
		Classifies gaze for the current frame based on the method specified in the config.
		This is only called when a trial is active and a face is detected.
		"""
		# 1. Check if we are within the stimulus period for the active trial.
		#    If not, there's nothing to classify, so we exit.
		is_stim_period = (current_frame_time_ms >= self.current_trial_data['start_time_ms'] and
		                  current_frame_time_ms < self.current_trial_data['stimulus_end_time_ms'])
		if not is_stim_period:
			self.gaze_on_stimulus_display_text = ""
			return

		# 2. This frame is valid for gaze processing. Increment the counter.
		self.current_trial_data['stimulus_frames_processed_gaze'] += 1

		gaze_on_stim_area_this_frame = False
		gaze_method = self.GAZE_CLASSIFICATION_METHOD

		# --- Main classification logic ---
		if gaze_method == "head_pose_only":
			if self.calibrated:
				pitch_ok = self.STIMULUS_PITCH_RANGE[0] <= self.adj_pitch <= self.STIMULUS_PITCH_RANGE[1]
				yaw_ok = self.STIMULUS_YAW_RANGE[0] <= self.adj_yaw <= self.STIMULUS_YAW_RANGE[1]
				if pitch_ok and yaw_ok:
					gaze_on_stim_area_this_frame = True

		elif gaze_method == "compensatory_gaze":
			if self.calibrated:
				# This method combines head yaw and eye dx sum.
				# First, check the simple vertical constraints (pitch and eye dy)
				pitch_ok = self.COMPENSATORY_HEAD_PITCH_RANGE[0] <= self.adj_pitch <= \
				           self.COMPENSATORY_HEAD_PITCH_RANGE[1]

				left_dy_ok = self.STIMULUS_LEFT_IRIS_DY_RANGE[0] <= self.l_dy <= self.STIMULUS_LEFT_IRIS_DY_RANGE[1]
				right_dy_ok = self.STIMULUS_RIGHT_IRIS_DY_RANGE[0] <= self.r_dy <= self.STIMULUS_RIGHT_IRIS_DY_RANGE[1]
				dy_ok = left_dy_ok and right_dy_ok

				if pitch_ok and dy_ok:
					head_yaw = self.adj_yaw
					eye_dx_sum = self.l_dx + self.r_dx

					# Condition 1: Head is centered, eyes are centered
					head_center_ok = self.COMPENSATORY_HEAD_YAW_RANGE_CENTER[0] <= head_yaw <= \
					                 self.COMPENSATORY_HEAD_YAW_RANGE_CENTER[1]
					eyes_center_ok = self.COMPENSATORY_EYE_SUM_RANGE_CENTER[0] <= eye_dx_sum <= \
					                 self.COMPENSATORY_EYE_SUM_RANGE_CENTER[1]

					# Condition 2: Head is turned left, eyes compensate to the right
					head_left_ok = self.COMPENSATORY_HEAD_YAW_RANGE_LEFT[0] <= head_yaw <= \
					               self.COMPENSATORY_HEAD_YAW_RANGE_LEFT[1]
					eyes_right_comp_ok = self.COMPENSATORY_EYE_SUM_RANGE_LEFT_TURN[0] <= eye_dx_sum <= \
					                     self.COMPENSATORY_EYE_SUM_RANGE_LEFT_TURN[1]

					# Condition 3: Head is turned right, eyes compensate to the left
					head_right_ok = self.COMPENSATORY_HEAD_YAW_RANGE_RIGHT[0] <= head_yaw <= \
					                self.COMPENSATORY_HEAD_YAW_RANGE_RIGHT[1]
					eyes_left_comp_ok = self.COMPENSATORY_EYE_SUM_RANGE_RIGHT_TURN[0] <= eye_dx_sum <= \
					                    self.COMPENSATORY_EYE_SUM_RANGE_RIGHT_TURN[1]

					if (head_center_ok and eyes_center_ok) or \
							(head_left_ok and eyes_right_comp_ok) or \
							(head_right_ok and eyes_left_comp_ok):
						gaze_on_stim_area_this_frame = True

		elif "eye_gaze" in gaze_method:  # Catches "eye_gaze_only" and "eye_gaze_with_head_filter"
			# --- A: Evaluate Eye Gaze component (common to both eye_gaze methods) ---
			# A.1: Vertical check (dy)
			left_dy_ok = self.STIMULUS_LEFT_IRIS_DY_RANGE[0] <= self.l_dy <= self.STIMULUS_LEFT_IRIS_DY_RANGE[1]
			right_dy_ok = self.STIMULUS_RIGHT_IRIS_DY_RANGE[0] <= self.r_dy <= self.STIMULUS_RIGHT_IRIS_DY_RANGE[1]
			dy_ok = left_dy_ok and right_dy_ok

			# A.2: Horizontal check (dx)
			dx_ok = False
			if self.GAZE_DX_SUM_THRESHOLD < 999:  # Correctly checks if the sum method is enabled
				# New, robust method: Check the sum of horizontal deviations
				dx_sum = self.l_dx + self.r_dx
				if abs(dx_sum) <= self.GAZE_DX_SUM_THRESHOLD:
					dx_ok = True
			else:
				# Fallback to original method: Check individual eye ranges
				left_dx_ok = self.STIMULUS_LEFT_IRIS_DX_RANGE[0] <= self.l_dx <= self.STIMULUS_LEFT_IRIS_DX_RANGE[1]
				right_dx_ok = self.STIMULUS_RIGHT_IRIS_DX_RANGE[0] <= self.r_dx <= self.STIMULUS_RIGHT_IRIS_DX_RANGE[1]
				if left_dx_ok and right_dx_ok:
					dx_ok = True

			eye_gaze_ok = dx_ok and dy_ok

			# --- B: Combine with head pose filter if needed ---
			if gaze_method == "eye_gaze_with_head_filter":
				if eye_gaze_ok and self.calibrated:
					# Use the wider filter ranges for head pose
					head_pitch_ok = self.HEAD_POSE_FILTER_PITCH_RANGE[0] <= self.adj_pitch <= \
					                self.HEAD_POSE_FILTER_PITCH_RANGE[1]
					head_yaw_ok = self.HEAD_POSE_FILTER_YAW_RANGE[0] <= self.adj_yaw <= self.HEAD_POSE_FILTER_YAW_RANGE[
						1]
					if head_pitch_ok and head_yaw_ok:
						gaze_on_stim_area_this_frame = True
			elif gaze_method == "eye_gaze_only":
				if eye_gaze_ok:
					gaze_on_stim_area_this_frame = True

		# --- Finalize and update state for this frame ---
		if gaze_on_stim_area_this_frame:
			self.current_trial_data['frames_on_stimulus_area'] += 1
			self.gaze_on_stimulus_display_text = "GAZE ON STIMULUS"
		else:
			self.gaze_on_stimulus_display_text = "GAZE OFF STIMULUS"

	def _log_frame_data(self, current_frame_time_ms, frame_count, landmarks: Optional[FaceLandmarks], img_w, img_h):
		current_log_timestamp = 0
		if not self.starting_timestamp:  # This case might not be hit if starting_timestamp is always set in __init__
			current_log_timestamp = int(time.time() * 1000)
		elif self.starting_timestamp:
			# Timestamps should be absolute from the original video start,
			# or relative to part start if we adjust self.starting_timestamp for part 2.
			# For now, let's keep it absolute from original video start.
			# The frame_count is also absolute.
			frame_increment_for_log = dt.timedelta(seconds=(1.0 / self.FPS if self.FPS > 0 else (1.0 / 30.0)))
			current_log_timestamp_dt = self.starting_timestamp + (frame_increment_for_log * frame_count)
			current_log_timestamp = int(current_log_timestamp_dt.strftime(self.TIMESTAMP_FORMAT))

		log_entry = [current_log_timestamp, frame_count,
		             self.l_cx, self.l_cy, self.r_cx, self.r_cy,
		             self.l_dx, self.l_dy, self.r_dx, self.r_dy,
		             self.TOTAL_BLINKS]  # TOTAL_BLINKS is per-part
		if self.ENABLE_HEAD_POSE:
			log_entry.extend([self.adj_pitch, self.adj_yaw, self.adj_roll])

		if self.LOG_ALL_FEATURES:
			if landmarks and landmarks.mp_multi_face_landmarks and landmarks.mp_multi_face_landmarks.multi_face_landmarks:
				lm_flat = []
				for p in landmarks.mp_multi_face_landmarks.multi_face_landmarks[0].landmark:
					lm_flat.extend([p.x * img_w, p.y * img_h])
					if self.LOG_Z_COORD: lm_flat.append(p.z)
				log_entry.extend(lm_flat)
			else:
				num_landmarks = 478 if self.USE_ATTENTION_MESH else 468
				num_coords = 3 if self.LOG_Z_COORD else 2
				log_entry.extend([0] * (num_landmarks * num_coords))
		self.csv_data.append(log_entry)

		if self.USE_SOCKET:
			packet = np.array([current_log_timestamp], dtype=np.int64).tobytes() + \
			         np.array([self.l_cx, self.l_cy, int(self.l_dx), int(self.l_dy)],
			                  dtype=np.int32).tobytes()
			self.socket.sendto(packet, self.SERVER_ADDRESS)
			if self.PRINT_DATA: self.logger.info(f'Sent UDP packet to {self.SERVER_ADDRESS}')

	def _draw_on_screen_data(self, frame, landmarks: Optional[FaceLandmarks], img_h, img_w, current_frame_time_ms):
		font_face = cv.FONT_HERSHEY_SIMPLEX
		font_scale_main = 0.55
		font_scale_small = 0.45
		font_thickness = 1
		line_h = 22
		text_color_green = (0, 255, 0)
		text_color_orange = (0, 165, 255)
		text_color_red = (0, 0, 255)
		text_color_magenta = (255, 0, 255)
		text_color_cyan = (255, 255, 0)
		text_color_yellow = (0, 255, 255)  # For trial result

		# Top-left column
		tl_x, y_pos = 10, 20
		# Display current part number
		cv.putText(frame, f"Part: {self.current_video_part}", (tl_x, y_pos), font_face, font_scale_small,
		           text_color_orange, font_thickness)
		y_pos += line_h - 5

		cv.putText(frame, f"Blinks: {self.TOTAL_BLINKS}", (tl_x, y_pos), font_face, font_scale_main, text_color_green,
		           font_thickness)
		y_pos += line_h
		if self.ENABLE_HEAD_POSE:
			# ... (rest of head pose display, unchanged)
			if self.calibrated:
				cv.putText(frame, f"Cal Pitch: {self.adj_pitch:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_green, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Cal Yaw: {self.adj_yaw:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_green, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Cal Roll: {self.adj_roll:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_green, font_thickness)
				y_pos += line_h
			else:
				cv.putText(frame, f"Raw Pitch: {self.smooth_pitch:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_orange, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Raw Yaw: {self.smooth_yaw:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_orange, font_thickness)
				y_pos += line_h
				cv.putText(frame, f"Raw Roll: {self.smooth_roll:.1f}", (tl_x, y_pos), font_face, font_scale_main,
				           text_color_orange, font_thickness)
				y_pos += line_h
				status_text = ""
				if not landmarks:
					status_text = "(No face for pose)"
				elif self.auto_calibrate_pending:
					status_text = "(Wait auto-calib)"
				elif not self.calibrated:
					status_text = "(Press 'c' to calib)"
				if status_text: cv.putText(frame, status_text, (tl_x, y_pos), font_face, font_scale_small,
				                           text_color_orange, font_thickness); y_pos += line_h - 5

		# Top-right column
		tr_y_pos, tr_x_anchor = 20, img_w - 10

		# Display the result of the last completed trial
		if self.ENABLE_VIDEO_TRIAL_DETECTION and self.last_trial_result_text:
			(w, _), _ = cv.getTextSize(self.last_trial_result_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, self.last_trial_result_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_main,
			           text_color_yellow, font_thickness)
			tr_y_pos += line_h

		if self.ENABLE_VIDEO_TRIAL_DETECTION and self.current_trial_data and self.current_trial_data['active']:
			trial_id = self.current_trial_data['id']
			stim_t_left = (self.current_trial_data['stimulus_end_time_ms'] - current_frame_time_ms) / 1000.0
			trial_t_left = (self.current_trial_data['trial_end_time_ms'] - current_frame_time_ms) / 1000.0
			trial_text = f"Trial {trial_id}" + (
				f" (Stim: {stim_t_left:.1f}s)" if stim_t_left > 0 else f" (Post: {trial_t_left:.1f}s)")
			(w, _), _ = cv.getTextSize(trial_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, trial_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_main, text_color_magenta,
			           font_thickness)
			tr_y_pos += line_h

		if self.face_looks_display_text:
			(w, _), _ = cv.getTextSize(self.face_looks_display_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, self.face_looks_display_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_main,
			           text_color_green, font_thickness)
			tr_y_pos += line_h

		if self.ENABLE_VIDEO_TRIAL_DETECTION:  # ROI Box and Text
			x_r, y_r, w_r, h_r = self.STIMULUS_ROI_COORDS
			cv.rectangle(frame, (x_r, y_r), (min(img_w, x_r + w_r), min(img_h, y_r + h_r)), text_color_cyan, 1)

			# --- THIS IS THE CORRECTED PART ---
			# It now uses roi_baseline_mean and roi_baseline_std_dev for the display.
			if self.roi_baseline_mean is not None:
				roi_base = f"{self.roi_baseline_mean:.1f} (SD:{self.roi_baseline_std_dev:.1f})"
			else:
				roi_base = "Wait"
			roi_text = f"ROI: {self.current_roi_brightness:.1f} (Base: {roi_base})"
			# --- END OF CORRECTION ---

			(w, _), _ = cv.getTextSize(roi_text, font_face, font_scale_small, font_thickness)
			cv.putText(frame, roi_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_cyan,
			           font_thickness)
			tr_y_pos += line_h

		if landmarks:
			# Always show eye gaze coordinates if a face is detected
			l_eye_text = f"L Eye D(xy): ({self.l_dx:.1f}, {self.l_dy:.1f})"
			r_eye_text = f"R Eye D(xy): ({self.r_dx:.1f}, {self.r_dy:.1f})"
			(w, _), _ = cv.getTextSize(l_eye_text, font_face, font_scale_small, font_thickness)
			cv.putText(frame, l_eye_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_cyan,
			           font_thickness)
			tr_y_pos += int(line_h * 0.8)
			(w, _), _ = cv.getTextSize(r_eye_text, font_face, font_scale_small, font_thickness)
			cv.putText(frame, r_eye_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_cyan,
			           font_thickness)
			tr_y_pos += int(line_h * 0.8)

			# --- ADDED FOR DEBUGGING ---
			# Calculate and display the sum of eye coordinates
			eye_dx_sum = self.l_dx + self.r_dx
			eye_dy_sum = self.l_dy + self.r_dy
			sum_text = f"Eye Sum D(xy): ({eye_dx_sum:.1f}, {eye_dy_sum:.1f})"
			(w, _), _ = cv.getTextSize(sum_text, font_face, font_scale_small, font_thickness)
			cv.putText(frame, sum_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small, text_color_yellow,
			           font_thickness)
			tr_y_pos += int(line_h * 0.8)
			# --- END OF ADDED CODE ---

			if self.is_looking_down_explicitly:
				down_text = "Eyes: Explicitly Down"
				(w, _), _ = cv.getTextSize(down_text, font_face, font_scale_small, font_thickness)
				cv.putText(frame, down_text, (tr_x_anchor - w, tr_y_pos), font_face, font_scale_small,
				           text_color_orange, font_thickness)
				tr_y_pos += int(line_h * 0.8)

		if self.gaze_on_stimulus_display_text:
			color = text_color_green if "ON" in self.gaze_on_stimulus_display_text else text_color_red
			(w, _), _ = cv.getTextSize(self.gaze_on_stimulus_display_text, font_face, font_scale_main, font_thickness)
			cv.putText(frame, self.gaze_on_stimulus_display_text, (img_w // 2 - w // 2, 20), font_face, font_scale_main,
			           color, font_thickness)

		# --- NEW: Tuning Mode Display ---
		if self.TUNING_MODE and self.calibrated:
			tuning_font_scale = 1.0
			tuning_font_thickness = 2
			pitch_text = f"Pitch: {self.adj_pitch:.1f}"
			yaw_text = f"Yaw:   {self.adj_yaw:.1f}"
			(w_p, h_p), _ = cv.getTextSize(pitch_text, font_face, tuning_font_scale, tuning_font_thickness)
			(w_y, h_y), _ = cv.getTextSize(yaw_text, font_face, tuning_font_scale, tuning_font_thickness)
			center_x = img_w // 2
			cv.putText(frame, pitch_text, (center_x - w_p // 2, img_h // 2 - h_p), font_face, tuning_font_scale, text_color_yellow, tuning_font_thickness)
			cv.putText(frame, yaw_text, (center_x - w_y // 2, img_h // 2 + h_y), font_face, tuning_font_scale, text_color_yellow, tuning_font_thickness)
		# --- End Tuning Mode Display ---

		cv.putText(frame, f'FPS: {self.FPS:.1f}', (tl_x, img_h - 10), font_face, font_scale_main, text_color_green,
		           font_thickness)

		if landmarks and landmarks.mp_multi_face_landmarks and landmarks.mp_multi_face_landmarks.multi_face_landmarks:
			face_landmarks_mp = landmarks.mp_multi_face_landmarks.multi_face_landmarks[0]
			if self.SHOW_ALL_FEATURES:
				mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks_mp,
				                          connections=mp_face_mesh.FACEMESH_TESSELATION,
				                          landmark_drawing_spec=drawing_spec,
				                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
			else:
				if self.l_cx != 0 or self.l_cy != 0:
					cv.circle(frame, (self.l_cx, self.l_cy), 2, (0, 255, 0), -1, cv.LINE_AA)
				if self.r_cx != 0 or self.r_cy != 0:
					cv.circle(frame, (self.r_cx, self.r_cy), 2, (0, 255, 0), -1, cv.LINE_AA)
				if self.ENABLE_HEAD_POSE and hasattr(self, '_indices_pose') and self.mesh_points is not None:
					for idx in self._indices_pose:
						if 0 <= idx < len(self.mesh_points): cv.circle(frame, self.mesh_points[idx], 2, (0, 0, 255), -1,
						                                               cv.LINE_AA)

	def _write_video_frame(self, frame):
		if self.out and self.out.isOpened():
			self.out.write(frame)
		elif self.out and not self.out.isOpened() and self.VIDEO_OUTPUT_BASE and self.frame_count % 100 == 0:
			if self.PRINT_DATA: self.logger.warning(f"VideoWriter for current part is not open.")

	def _handle_key_presses(self, key_pressed):
		if key_pressed == ord('q'):
			if self.PRINT_DATA: self.logger.info("Exiting program...")
			return True
		return False

	def _finalize_part(self, part_suffix=""):
		"""Finalizes writing logs and video for the current part."""
		if self.PRINT_DATA: self.logger.info(f"Finalizing data for Part {self.current_video_part} with suffix '{part_suffix}'...")
		if self.out and self.out.isOpened():
			self.out.release()
			self.out = None  # Important to set to None
			if self.PRINT_DATA: self.logger.info(f"Released video writer for Part {self.current_video_part}.")

		self._save_trial_summary(part_suffix)
		self._save_main_log(part_suffix)

	def _prepare_for_next_part(self, split_time_ms_absolute):
		"""Resets state for processing the next part of the video."""
		if self.PRINT_DATA: print(
			self.logger.info(f"Preparing for Part {self.current_video_part + 1} starting around {split_time_ms_absolute}ms (original time)..."))
		self.current_video_part += 1

		# Reset data accumulators
		self.csv_data = []
		self.all_trials_summary = []
		self.trial_counter = 0
		self.TOTAL_BLINKS = 0
		self.EYES_BLINK_FRAME_COUNTER = 0

		# Reset trial detection state
		self.current_trial_data = None
		# Adjust last_trial_end_time_ms to be relative to the new part's start,
		# but using the absolute timeline for comparison with current_frame_time_ms.
		# Effectively, a trial can start soon after the split if conditions are met.
		self.last_trial_end_time_ms = split_time_ms_absolute - self.MIN_INTER_TRIAL_INTERVAL_MS
		self.roi_brightness_samples = []
		self.roi_baseline_mean = None
		self.roi_baseline_std_dev = None
		if hasattr(self, 'last_trial_result_text'):
			self.last_trial_result_text = ""

		# Reset angle buffer for smoothing
		self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)

		# Re-initialize video output for the new part
		if self.VIDEO_OUTPUT_BASE:
			self.out = self.init_video_output(part_suffix=self.output_suffix_part2)
			if self.out is None and self.PRINT_DATA:
				self.logger.warning(f"Failed to initialize video output for Part {self.current_video_part}")

		# Calibration (self.calibrated, self.initial_pitch, etc.) persists.
		# self.auto_calibrate_pending also persists (or could be reset if desired).
		# self.starting_timestamp (original video start) persists for absolute frame time calculation.

		if self.PRINT_DATA: self.logger.info(f"State reset for Part {self.current_video_part}.")

	def _save_trial_summary(self, part_suffix=""):
		if not (self.ENABLE_VIDEO_TRIAL_DETECTION and self.all_trials_summary):
			if self.ENABLE_VIDEO_TRIAL_DETECTION and self.PRINT_DATA: self.logger.info(
				f"No trials to summarize for current part{part_suffix}.")
			return

		if self.current_trial_data and self.current_trial_data['active']:
			if self.PRINT_DATA: self.logger.info(
				f"Finalizing active trial {self.current_trial_data['id']} on exit/split (Part {self.current_video_part}).")
			if self.current_trial_data['stimulus_frames_processed_gaze'] > 0:
				perc_on_stim = (self.current_trial_data['frames_on_stimulus_area'] /
				                self.current_trial_data['stimulus_frames_processed_gaze']) * 100
				# If the percentage is below the threshold, classify as 'away' (2).
				# Otherwise, it remains the default 'looked' (1).
				if perc_on_stim < self.LOOK_TO_STIMULUS_THRESHOLD_PERCENT:
					self.current_trial_data['looked_final'] = 2
			self.all_trials_summary.append(self.current_trial_data.copy())
			self.current_trial_data = None  # Clear after adding

		ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
		folder = self.TRACKING_DATA_LOG_FOLDER or "."
		subj = self.subject_id or 'NA'

		base_filename = self.OUTPUT_TRIAL_SUMMARY_FILENAME_PREFIX or "trial_summary_"
		summary_fn = os.path.join(folder, f"{subj}_{self.session}_{base_filename.replace('.csv', '')}{part_suffix}_{ts_str}.csv")

		os.makedirs(os.path.dirname(summary_fn), exist_ok=True)

		with open(summary_fn, "w", newline="") as f:
			writer = csv.writer(f)
			headers = ['trial_id', 'start_time_ms', 'stimulus_end_time_ms', 'trial_end_time_ms',
			           'stimulus_frames_processed_gaze', 'frames_on_stimulus_area', 'looked_at_stimulus']
			writer.writerow(headers)
			for trial_sum in self.all_trials_summary:
				writer.writerow([
					trial_sum['id'], trial_sum['start_time_ms'], trial_sum['stimulus_end_time_ms'],
					trial_sum['trial_end_time_ms'], trial_sum['stimulus_frames_processed_gaze'],
					trial_sum['frames_on_stimulus_area'], trial_sum['looked_final']])
		if self.PRINT_DATA: self.logger.info(f"Trial summary saved: {summary_fn}")

	def _save_main_log(self, part_suffix=""):
		if not (self.LOG_DATA and self.csv_data):
			if self.LOG_DATA and self.PRINT_DATA: self.logger.info(f"No main log data to write for current part{part_suffix}.")
			return

		if self.PRINT_DATA: self.logger.info(f"Writing main log data for current part{part_suffix} to CSV...")
		ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
		folder = self.TRACKING_DATA_LOG_FOLDER or "."
		subj = self.subject_id or 'NA'

		base_filename = getattr(self, "OUTPUT_MAIN_LOG_FILENAME_PREFIX", "eye_tracking_log_")
		# --- IMPROVEMENT: Embed FPS directly into the filename for easier parsing later ---
		fps_str = f"{self.FPS:.1f}fps"
		csv_fn = os.path.join(folder, f"{subj}_{self.session}_{base_filename.replace('.csv', '')}{part_suffix}_{fps_str}_{ts_str}.csv")
		os.makedirs(os.path.dirname(csv_fn), exist_ok=True)

		# Padding logic - consider if this is still desired per part, or only for the whole video.
		# If per part, total_frames would need to be for that part.
		# For now, let's assume padding is not strictly needed if splitting, or needs more complex handling.
		# The current padding uses self.total_frames which is for the whole video.
		# This might lead to excessive padding for part 1.
		# Let's simplify: no padding if splitting, or user ensures total_frames is for the first part if they want padding for it.

		# if self.total_frames > 0 and len(self.csv_data) < self.total_frames and self.current_video_part == 1: # Example: Pad only part 1
		# ... padding logic ...

		with open(csv_fn, "w", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(self.column_names)
			writer.writerows(self.csv_data)
		if self.PRINT_DATA: self.logger.info(f"Main log data saved: {csv_fn}")

	def _reset_analysis_state(self):
		"""Resets all state variables required for a fresh analysis pass."""
		if self.PRINT_DATA:
			self.logger.info("Resetting tracker state for main analysis pass...")

		# Re-initialize video capture to start from the beginning
		self.cap = self.init_video_input()

		# Re-initialize video writer for the first part
		current_output_suffix = self.output_suffix_part1 if self.split_at_ms is not None else ""
		if self.VIDEO_OUTPUT_BASE:
			self.out = self.init_video_output(part_suffix=current_output_suffix)
		else:
			self.out = None

		# Reset data accumulators and counters
		self.TOTAL_BLINKS = 0
		self.EYES_BLINK_FRAME_COUNTER = 0
		self.csv_data = []
		self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)

		# Reset video splitting state
		self.current_video_part = 1
		self.split_triggered_and_finalized = False

		# Reset trial detection state
		if self.ENABLE_VIDEO_TRIAL_DETECTION:
			self.trial_counter = 0
			self.current_trial_data = None
			self.all_trials_summary = []
			self.last_trial_end_time_ms = -self.MIN_INTER_TRIAL_INTERVAL_MS
			self.roi_brightness_samples = []
			self.roi_baseline_mean = None
			self.roi_baseline_std_dev = None
			self.last_trial_result_text = ""

	def _reset_for_main_pass(self):
		"""
		Resets the tracker state after a pre-analysis pass (like find_first_stimulus_onset)
		but preserves the results of calibration and ROI detection.
		"""
		if self.cap and self.cap.isOpened():
			self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

		# Reset only the data accumulators and trial-specific state
		self.TOTAL_BLINKS = 0
		self.EYES_BLINK_FRAME_COUNTER = 0
		self.csv_data = []
		self.angle_buffer = AngleBuffer(size=self.MOVING_AVERAGE_WINDOW)

		if self.ENABLE_VIDEO_TRIAL_DETECTION:
			self.trial_counter = 0
			self.current_trial_data = None
			self.all_trials_summary = []
			self.last_trial_end_time_ms = -self.MIN_INTER_TRIAL_INTERVAL_MS
			self.roi_brightness_samples = []
			self.roi_baseline_mean = None

	def _execute_calibration_pass(self):
		"""
		First pass over the video to collect head pose data and perform clustering calibration.
		Returns True on success, False on failure.
		"""
		if not self.cap or not self.cap.isOpened():
			self.logger.error("Video capture not open for calibration pass.")
			return False

		calib_duration_sec = getattr(self, "CLUSTERING_CALIB_DURATION_SECONDS", 30)
		frame_limit = int(calib_duration_sec * self.FPS)

		frame_num = 0
		while self.cap.isOpened() and frame_num < frame_limit:			
			frame, img_h, img_w, ret = self._get_and_preprocess_frame()
			if not ret:
				if self.PRINT_DATA: self.logger.info("Video ended before calibration period finished.")
				break

			# We only need to run face mesh and head pose estimation
			landmarks = self.detector.detect(frame)

			if landmarks:
				# _process_head_pose will see that clustering is pending and append samples
				self._process_head_pose(landmarks, img_h, img_w, key_pressed=-1)

			frame_num += 1
			if self.PRINT_DATA and frame_num % int(self.FPS or 30) == 0:
				self.logger.info(f"  Calibration Pass: Processed {frame_num}/{frame_limit} frames...")

		# If the video was shorter than the calibration period, we might need to trigger calibration manually
		if not self.calibrated and len(self.clustering_calib_all_samples) > 0:
			self._perform_clustering_calibration()

		# Release the capture object; it will be re-initialized for the main pass
		self.cap.release()

		# After calibration, we are no longer pending auto-calibration
		self.auto_calibrate_pending = False

		return self.calibrated

	def _cleanup(self, finalize_data=True):
		"""Releases resources and finalizes data saving."""
		if self.cap and self.cap.isOpened():
			self.cap.release()

		if finalize_data:
			# Finalize the last part being processed
			final_suffix = self.output_suffix_part2 if self.split_triggered_and_finalized else \
				(self.output_suffix_part1 if self.split_at_ms is not None else "")
			self._finalize_part(part_suffix=final_suffix)

		# Wrap GUI cleanup in a broad try/except to prevent crashes on exit
		# if running in a headless environment or if OpenCV is misconfigured.
		try:
			cv.destroyAllWindows()
		except Exception:
			pass

		if hasattr(self, 'socket') and self.socket:
			self.socket.close()
		if self.PRINT_DATA: self.logger.info("Program exited.")

	def run(self):
		"""
		Main entry point. Orchestrates calibration and analysis passes.
		"""
		# --- NEW: Check if pre-analysis passes were already run (e.g., by find_first_stimulus_onset) ---
		# If the tracker is already calibrated, we can skip the entire setup sequence.
		if self.calibrated:
			self.logger.info("--- Tracker is already primed. Skipping setup passes and starting main analysis. ---")
		else:
			# --- Standard Setup Execution Path for a fresh instance ---
			if getattr(self, 'STIMULUS_ROI_METHOD', 'static') == 'dynamic':
				self.logger.info("--- Starting Pass 1: Dynamic ROI Detection ---")
				if not self._find_dynamic_roi():
					self.logger.error("Dynamic ROI detection failed. Aborting analysis.")
					self._cleanup(finalize_data=False)
					return
				self.logger.info("--- Dynamic ROI Detection Complete ---")

			if self.CALIBRATION_METHOD == 'clustering':
				self.logger.info("--- Starting Pass 2: Calibration Data Collection ---")
				if not self._execute_calibration_pass():
					self.logger.error("Clustering calibration failed. A face may not have been consistently visible.")
					self.logger.error("Aborting analysis.")
					self._cleanup(finalize_data=False)
					return
				self.logger.info("--- Calibration Complete. Starting Final Pass: Full Analysis ---")
				self._reset_analysis_state()  # This is the problematic call for the re-init workflow
			else:
				self.logger.info("--- Starting Single-Pass Analysis ---")

		self.logger.info("="*50)
		# =================================================================================
		# --- Main Analysis Loop ---
		# =================================================================================
		self.frame_count = -1
		try:
			while True:
				self.frame_count += 1
				self._reset_per_frame_state()

				# --- NEW: Frame Skipping Logic ---
				if self.frame_count > 0 and self.FRAME_SKIP > 1 and (self.frame_count % self.FRAME_SKIP) != 0:
					ret = self.cap.grab() # Efficiently skip frame without decoding
					if not ret: break
					continue # Go to the next iteration of the loop

				frame, img_h, img_w, ret = self._get_and_preprocess_frame()
				if not ret:
					break

				current_frame_time_ms = int(self.frame_count * (1000.0 / self.FPS))
				if self.PRINT_DATA and self.frame_count % 60 == 0:
					self.logger.info(
						f"Frame Nr.: {self.frame_count}, Time: {current_frame_time_ms}ms (Part {self.current_video_part})")

				# --- Video Splitting Logic ---
				if self.split_at_ms is not None and \
						not self.split_triggered_and_finalized and \
						current_frame_time_ms >= self.split_at_ms:
					if self.PRINT_DATA: self.logger.info(f"Split point reached at {current_frame_time_ms}ms. Finalizing Part 1.")
					self._finalize_part(part_suffix=self.output_suffix_part1)
					self._prepare_for_next_part(current_frame_time_ms)
					self.split_triggered_and_finalized = True
				# --- End Splitting Logic ---

				key_pressed = cv.waitKey(1) & 0xFF

				landmarks = self.detector.detect(frame)
				self.mesh_points = landmarks.raw_mesh_points_2d if landmarks else None

				if landmarks:
					self._extract_eye_features(landmarks)
					self.is_looking_down_explicitly = self._check_downward_look()
					if landmarks.raw_mesh_points_3d is not None:
						self._update_blink_count(landmarks.raw_mesh_points_3d)

					if self.ENABLE_HEAD_POSE:
						# For clustering, this now uses the pre-calibrated baseline.
						# For other methods, it performs live calibration if needed.
						self._process_head_pose(landmarks, img_h, img_w, key_pressed)
						self.face_looks_display_text = self._get_face_looks_text()

				if self.ENABLE_VIDEO_TRIAL_DETECTION:
					# If EEG onsets are provided, use them. Otherwise, fall back to video detection. (Bypassed in tuning mode)
					if self.eeg_trial_onsets_ms:
						self._check_for_eeg_trial_onset(current_frame_time_ms)
					else:
						self.current_roi_brightness = self._calculate_roi_brightness(frame, img_h, img_w)
						self._update_trial_state(current_frame_time_ms)

					# --- NEW: Centralized Trial Ending and Gaze Classification ---
					# End the trial if its time is up. This MUST run for both EEG and video trials.
					self._end_active_trial_if_needed(current_frame_time_ms)

					# If a trial is active and we are NOT in tuning mode, classify gaze.
					if not self.TUNING_MODE:
						if self.current_trial_data and self.current_trial_data['active'] and \
								landmarks:
							self._classify_gaze_for_current_trial(current_frame_time_ms)

				if self.LOG_DATA:
					self._log_frame_data(current_frame_time_ms, self.frame_count, landmarks, img_w, img_h)

				if self.SHOW_ON_SCREEN_DATA:
					self._draw_on_screen_data(frame, landmarks, img_h, img_w, current_frame_time_ms)
					cv.imshow("Eye Tracking", frame)

				self._write_video_frame(frame)

				if self._handle_key_presses(key_pressed):
					break
		except Exception as e:
			self.logger.error(f"An error occurred in the main run loop: {e}", exc_info=True)
			import traceback
			traceback.print_exc()
		finally:
			self._cleanup()


if __name__ == "__main__":
	config_path = "config.yaml"  # Ensure this path is correct
	try:
		# Example:
		# tracker = ocapi(
		# subject_id="test_split",
		# config_file_path=config_path,
		# VIDEO_INPUT="path/to/your/long_video.mp4",
		# VIDEO_OUTPUT="output/test_split_processed.mp4", # Base name for output video
		# TRACKING_DATA_LOG_FOLDER="output/logs_test_split"
		# )
		tracker = Ocapi(config_file_path=config_path)  # For default testing from config
		tracker.run()
	except Exception as e:
		# If the logger hasn't been set up, this will print to console.
		logging.basicConfig()
		logging.critical(f"Failed to initialize or run Ocapi: {e}", exc_info=True)
		import traceback
		traceback.print_exc()
