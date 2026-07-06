#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocapi.ocapi import Ocapi

def run_test():
    print("=" * 80)
    print("               TESTING TEMPORAL DIFFERENCE TRIAL DETECTION")
    print("=" * 80)

    # Path configuration
    video_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/all_videos_combined"
    eeg_base_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/input"
    subject_id = "SCS062"
    session = "A"
    subject_sess = f"{subject_id}_{session}"

    video_path = os.path.join(video_dir, f"{subject_sess}_Video.mp4")
    eeg_path = os.path.join(eeg_base_dir, subject_id, session)

    found_headers = glob.glob(os.path.join(eeg_path, f"{subject_sess}*.vhdr"))
    if not found_headers:
        print(f"Error: EEG header file not found in {eeg_path}")
        return
    eeg_header_file = found_headers[0]

    found_markers = glob.glob(os.path.join(eeg_path, f"{subject_sess}*.vmrk"))
    if not found_markers:
        print(f"Error: EEG marker file not found in {eeg_path}")
        return
    eeg_marker_file = found_markers[0]

    print(f"Video file: {video_path}")
    print(f"EEG Marker file: {eeg_marker_file}")

    # 1. Initialize Ocapi
    print("\nInitializing Ocapi tracker...")
    tracker = Ocapi(
        subject_id=subject_id,
        session=session,
        config_file_path="config.yml",
        WEBCAM=None,
        VIDEO_INPUT=video_path,
        VIDEO_OUTPUT="validation_logs/SCS062_A_Video_marked.mp4",
        TRACKING_DATA_LOG_FOLDER="validation_logs"
    )
    # Optimize execution for speed: skip head calibration, skip logging, and skip landmarks detection
    tracker.SHOW_ON_SCREEN_DATA = False
    tracker.skip_video_decoding = False
    tracker.CALIBRATION_METHOD = "none"
    tracker.calibrated = True  # Avoid searching or doing calibration
    tracker.LOG_DATA = False
    tracker.total_frames = 18000  # Limit to 5 minutes
    tracker.detector.detect = lambda frame: None  # Mock detector to bypass heavy computation

    # 2. Load EEG triggers to get ground truth
    print("\nSynchronizing with EEG data for ground truth...")
    tracker.sync_with_eeg_and_set_onsets(
        header_file=eeg_header_file,
        marker_file=eeg_marker_file,
        stimulus_description="Stimulus",
        events_of_interest=["S 21", "S 22", "S 23", "S 24"]
    )
    ground_truth_onsets = list(tracker.eeg_trial_onsets_ms)
    print(f"Found {len(ground_truth_onsets)} ground-truth trials from EEG.")

    # 3. Clear EEG triggers to force video-based onset detection
    tracker.eeg_trial_onsets_ms = None

    # 4. Run tracking
    print(f"\nRunning tracking analysis with method: {tracker.TRIAL_ONSET_DETECTION_METHOD}...")
    print(f"Threshold STD DEV Multiplier: {getattr(tracker, 'ROI_DIFF_STD_DEV_THRESHOLD', 8.0)}")
    tracker.run()

    # 5. Extract detected trials
    detected_onsets = [t['start_time_ms'] for t in tracker.all_trials_summary]
    print(f"\nDetected {len(detected_onsets)} trials from video recordings.")

    # 6. Evaluation
    tolerance_ms = 1000.0  # Max latency to consider a match
    matched_gt = []
    matched_det = []
    latencies = []

    for gt in ground_truth_onsets:
        # Find closest detected onset within tolerance
        diffs = [abs(det - gt) for det in detected_onsets]
        if diffs:
            min_idx = np.argmin(diffs)
            min_diff = diffs[min_idx]
            if min_diff <= tolerance_ms:
                matched_gt.append(gt)
                matched_det.append(detected_onsets[min_idx])
                latencies.append(detected_onsets[min_idx] - gt)

    matched_count = len(matched_gt)
    missed_count = len(ground_truth_onsets) - matched_count
    false_positives = len(detected_onsets) - matched_count

    precision = matched_count / len(detected_onsets) if detected_onsets else 0
    recall = matched_count / len(ground_truth_onsets) if ground_truth_onsets else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 50)
    print("                 DETECTION RESULTS")
    print("=" * 50)
    print(f"Total Ground-Truth EEG Trials: {len(ground_truth_onsets)}")
    print(f"Total Video-Detected Trials:   {len(detected_onsets)}")
    print(f"Matched Trials (within {tolerance_ms/1000:.1f}s): {matched_count}")
    print(f"Missed EEG Trials:             {missed_count}")
    print(f"False Positive Detections:     {false_positives}")
    print("-" * 50)
    print(f"Precision:                     {precision:.4f}")
    print(f"Recall:                        {recall:.4f}")
    print(f"F1-Score:                      {f1:.4f}")
    if latencies:
        print(f"Mean Detection Latency:        {np.mean(latencies):.2f} ms")
        print(f"Latency Std Dev:               {np.std(latencies):.2f} ms")
    print("=" * 50)

if __name__ == '__main__':
    run_test()
