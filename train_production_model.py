#!/usr/bin/env python3
"""
train_production_model.py

Trains the final production gaze classifier using the optimized workflow:
- 24 gaze-only features mapped to the 54-feature production layout
- Consensus training labels (only agreed trials)
- Random oversampling for class imbalance
- Saccade latency offset (150ms)
- ExtraTrees Classifier (tuned parameters)
- Probability threshold set to 0.40 for optimized EEG trial retention

Saves the output to ocapi/models/gaze_classifier_gb.pkl.
"""
import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

project_root = os.path.dirname(os.path.abspath(__file__))
rater1_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_1"
rater2_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_2"
log_dir = os.path.join(project_root, "validation_logs")
model_dir = os.path.join(project_root, "ocapi", "models")
model_path = os.path.join(model_dir, "gaze_classifier_gb.pkl")
backup_path = os.path.join(model_dir, "gaze_classifier_gb_backup.pkl")

def discover_subjects():
    subjects = {}
    tracking_files = sorted(glob.glob(os.path.join(log_dir, "*_eye_tracking_log_*.csv")))
    for f in tracking_files:
        basename = os.path.basename(f)
        match = re.match(r"([A-Z]+\d+_[A-Z])_eye_tracking_log_", basename)
        if match:
            subj = match.group(1)
            subjects.setdefault(subj, {})["tracking_log"] = basename
    
    trial_files = sorted(glob.glob(os.path.join(log_dir, "*_trial_summary_*.csv")))
    for f in trial_files:
        basename = os.path.basename(f)
        match = re.match(r"([A-Z]+\d+_[A-Z])_trial_summary_", basename)
        if match:
            subj = match.group(1)
            subjects.setdefault(subj, {})["trial_summary"] = basename
    
    valid = {}
    for subj, cfg in subjects.items():
        if "tracking_log" in cfg and "trial_summary" in cfg:
            cfg["stim_duration_ms"] = 800
            valid[subj] = cfg
    return valid

def load_human_labels(subject_sess, rater_dir, col=0):
    matches = glob.glob(os.path.join(rater_dir, f"{subject_sess}*"))
    if not matches:
        return None
    df = pd.read_excel(matches[0], header=None)
    labels = df.iloc[:, col].replace(0, 2).tolist()
    return [int(l) for l in labels if not pd.isna(l)]

def stats(arr):
    if len(arr) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array(arr, dtype=np.float32)
    return [float(np.mean(arr)), float(np.std(arr)),
            float(np.percentile(arr, 10)), float(np.percentile(arr, 90))]

def extract_production_features(tracking_log_path, trial_summary_path, latency_offset_ms=150, stim_duration_ms=800):
    """
    Extracts all 54 features per trial exactly as structured in ocapi.py
    """
    df = pd.read_csv(tracking_log_path)
    df.columns = df.columns.str.strip()
    
    col_map = {
        'Timestamp (ms)': 'timestamp_ms',
        'Frame Nr': 'frame_nr',
        'Trial ID': 'trial_id',
        'Left Eye Center X': 'l_cx',
        'Left Eye Center Y': 'l_cy',
        'Right Eye Center X': 'r_cx',
        'Right Eye Center Y': 'r_cy',
        'Left Iris Relative Pos Dx': 'l_dx',
        'Left Iris Relative Pos Dy': 'l_dy',
        'Right Iris Relative Pos Dx': 'r_dx',
        'Right Iris Relative Pos Dy': 'r_dy',
        'Total Blink Count': 'blinks',
        'Pitch': 'pitch',
        'Yaw': 'yaw',
        'Roll': 'roll',
    }
    df = df.rename(columns=col_map)
    for col in col_map.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    df['pitch_diff'] = df['pitch'].diff().fillna(0.0)
    df['yaw_diff'] = df['yaw'].diff().fillna(0.0)
    df['l_dx_diff'] = df['l_dx'].diff().fillna(0.0)
    df['r_dx_diff'] = df['r_dx'].diff().fillna(0.0)
    df['vergence'] = df['l_dx'] - df['r_dx']
    
    ts = pd.read_csv(trial_summary_path)
    ts.columns = ts.columns.str.strip()
    
    fps = 60.0
    frame_dur_ms = 1000.0 / fps
    feature_rows = []
    
    for _, trial_row in ts.iterrows():
        tid = int(trial_row['trial_id'])
        stim_start = float(trial_row['start_time_ms'])
        
        window = df[df['trial_id'] == tid]
        n_frames = len(window)
        
        if n_frames == 0:
            approx_start_frame = int(stim_start / frame_dur_ms)
            approx_end_frame = int((stim_start + stim_duration_ms) / frame_dur_ms)
            window = df[(df['frame_nr'] >= approx_start_frame) & (df['frame_nr'] < approx_end_frame)]
            
        n_frames_total = len(window)
        if n_frames_total == 0:
            feat = [0.0] * 54
        else:
            # Apply saccade latency offset
            if latency_offset_ms > 0:
                frame_times = window['frame_nr'] * (1000.0 / fps)
                window = window[(frame_times - stim_start) >= latency_offset_ms]
                
            n_frames_after_offset = len(window)
            if n_frames_after_offset == 0:
                feat = [0.0] * 54
            else:
                has_face = ((window['l_cx'] != 0) | (window['r_cx'] != 0)).values.astype(bool)
                n_face = has_face.sum()
                face_frac = n_face / n_frames_after_offset
                
                w = window[has_face]
                
                pitch_stats = stats(w['pitch'].values)
                yaw_stats = stats(w['yaw'].values)
                l_dx_stats = stats(w['l_dx'].values)
                l_dy_stats = stats(w['l_dy'].values)
                r_dx_stats = stats(w['r_dx'].values)
                r_dy_stats = stats(w['r_dy'].values)
                
                pitch_diff_stats = stats(w['pitch_diff'].values)
                yaw_diff_stats = stats(w['yaw_diff'].values)
                l_dx_diff_stats = stats(w['l_dx_diff'].values)
                r_dx_diff_stats = stats(w['r_dx_diff'].values)
                
                vergence_stats = stats(w['vergence'].values)
                
                if len(w) > 0:
                    eye_sum = (w['l_dx'] + w['r_dx']).values
                    eye_sum_stats = stats(eye_sum)
                    head_centered_frac = float(((np.abs(w['pitch'].values) < 20) & (np.abs(w['yaw'].values) < 20)).mean())
                    eye_centered_frac = float((np.abs(eye_sum) < 5).mean())
                else:
                    eye_sum_stats = [0.0] * 4
                    head_centered_frac = 0.0
                    eye_centered_frac = 0.0
                    
                if len(window) > 1:
                    blink_diff = np.diff(window['blinks'].values, prepend=window['blinks'].values[0])
                    blinks_frac = float(np.mean((blink_diff > 0) | (~has_face)))
                else:
                    blinks_frac = 0.0
                    
                feat = (
                    [n_face, face_frac, float(n_frames_total)]
                    + pitch_stats
                    + yaw_stats
                    + l_dx_stats
                    + l_dy_stats
                    + r_dx_stats
                    + r_dy_stats
                    + eye_sum_stats
                    + pitch_diff_stats
                    + yaw_diff_stats
                    + l_dx_diff_stats
                    + r_dx_diff_stats
                    + vergence_stats
                    + [head_centered_frac, eye_centered_frac, blinks_frac]
                )
        feature_rows.append(feat)
        
    X = np.array(feature_rows, dtype=np.float32)
    for col_i in range(X.shape[1]):
        col_vals = X[:, col_i]
        nan_mask = np.isnan(col_vals)
        if nan_mask.all():
            X[:, col_i] = 0.0
        elif nan_mask.any():
            X[nan_mask, col_i] = np.nanmean(col_vals)
            
    return X

def random_oversample(X, y):
    np.random.seed(42)
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    X_resampled, y_resampled = [], []
    for cls in unique:
        cls_mask = y == cls
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]
        idx = np.random.choice(len(X_cls), size=max_count, replace=True)
        X_resampled.append(X_cls[idx])
        y_resampled.append(y_cls[idx])
    return np.vstack(X_resampled), np.concatenate(y_resampled)

def robust_subject_normalize(subject_X_dict):
    norm_X = {}
    for subj, X in subject_X_dict.items():
        X_norm = np.copy(X)
        for col in range(X.shape[1]):
            col_vals = X[:, col]
            median = np.median(col_vals)
            q75, q25 = np.percentile(col_vals, [75, 25])
            iqr = q75 - q25
            X_norm[:, col] = (col_vals - median) / iqr if iqr > 0 else col_vals - median
        norm_X[subj] = X_norm
    return norm_X

def main():
    print("=" * 70)
    print("  TRAINING OCAPI PRODUCTION CLASSIFIER")
    print("=" * 70)
    
    subjects_config = discover_subjects()
    subject_list = []
    
    for subj in sorted(subjects_config.keys()):
        y_r1 = load_human_labels(subj, rater1_dir)
        if y_r1 is not None:
            subject_list.append(subj)
            
    all_y_r1 = {}
    all_y_r2 = {}
    raw_X = {}
    
    print("Extracting features from all subjects...")
    for subj in subject_list:
        cfg = subjects_config[subj]
        y_r1 = load_human_labels(subj, rater1_dir)
        y_r2 = load_human_labels(subj, rater2_dir)
        
        X = extract_production_features(
            os.path.join(log_dir, cfg["tracking_log"]), 
            os.path.join(log_dir, cfg["trial_summary"]),
            latency_offset_ms=150,
            stim_duration_ms=cfg["stim_duration_ms"]
        )
        
        n = min(len(X), len(y_r1))
        raw_X[subj] = X[:n]
        all_y_r1[subj] = np.array(y_r1)[:n]
        
        if y_r2:
            y_r2_aligned = y_r2[:min(len(y_r2), n)]
            if len(y_r2_aligned) < n:
                y_r2_aligned = y_r2_aligned + [np.nan] * (n - len(y_r2_aligned))
            all_y_r2[subj] = np.array(y_r2_aligned)
        else:
            all_y_r2[subj] = None
            
    # Apply subject-wise Z-scoring normalization
    norm_X = robust_subject_normalize(raw_X)
    
    # ── Map the 24 gaze-only features in the 54 production layout ──
    # [0, 1, 2] = face metrics
    # [11:31] = gaze stats (l_dx, l_dy, r_dx, r_dy, eye_sum)
    # [52] = eye_centered_frac
    GAZE_ONLY_INDICES = [0, 1, 2] + list(range(11, 31)) + [52]
    
    X_list = []
    y_list = []
    
    # Build the full training set using only agreed (consensus) trials
    for subj in subject_list:
        X_s = norm_X[subj][:, GAZE_ONLY_INDICES]
        y_s_r1 = all_y_r1[subj]
        y_s_r2 = all_y_r2[subj]
        
        if y_s_r2 is not None:
            valid = ~np.isnan(y_s_r1.astype(float)) & ~np.isnan(y_s_r2.astype(float))
            agree = (y_s_r1 == y_s_r2) & valid
            if agree.sum() > 0:
                X_list.append(X_s[agree])
                y_list.append(y_s_r1[agree])
                
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)
    
    # Oversample minority class
    X_train, y_train = random_oversample(X_train, y_train)
    
    # Fit the scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Train the tuned ExtraTrees classifier
    print(f"Training ExtraTrees model on {len(X_train_s)} trials...")
    clf = ExtraTreesClassifier(
        n_estimators=300, 
        min_samples_leaf=4, 
        max_depth=12, 
        max_features="log2", 
        random_state=42, 
        n_jobs=1
    )
    clf.fit(X_train_s, y_train)
    
    # Create the model dict containing scaler, classifier, feature indices, threshold, and offset
    model_data = {
        "classifier": clf,
        "scaler": scaler,
        "feature_indices": GAZE_ONLY_INDICES,
        "threshold": 0.40,
        "latency_offset_ms": 150
    }
    
    # Save model and keep backup
    if os.path.exists(model_path):
        print(f"Creating backup of existing model at {backup_path}...")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(model_path, backup_path)
        
    print(f"Saving new model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print("Production model successfully trained, optimized, and saved.")

if __name__ == "__main__":
    main()
