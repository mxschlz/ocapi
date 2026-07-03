#!/usr/bin/env python3
"""
ml_gaze_classifier.py

Trains a gradient-boosted classifier on per-trial features extracted from
OCAPI eye-tracking logs, with leave-one-subject-out cross-validation.

Features per trial:
  - Head pose: mean/std/p10/p90 of pitch, yaw during stimulus window
  - Eye gaze: mean/std/p10/p90 of l_dx, l_dy, r_dx, r_dy (calibrated)
  - Fraction of frames with head in a rough "forward" range
  - Number of frames with valid face detection

Labels: human rater 1 coding (1=looked, 2=away)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

project_root = os.path.dirname(os.path.abspath(__file__))
rater1_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_1"
rater2_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_2"
log_dir = os.path.join(project_root, "validation_logs")

# Subjects and their most recent tracking logs (from the head_pose_only run - task-336)
SUBJECTS = {
    "SCS048_A": {
        "tracking_log": "SCS048_A_eye_tracking_log_60.0fps_20260702_135354.csv",
        "trial_summary": "SCS048_A_trial_summary_20260702_135354.csv",
        "stim_duration_ms": 800,
    },
    "SCS062_A": {
        "tracking_log": "SCS062_A_eye_tracking_log_60.0fps_20260702_135722.csv",
        "trial_summary": "SCS062_A_trial_summary_20260702_135722.csv",
        "stim_duration_ms": 800,
    },
    "SCS063_A": {
        "tracking_log": "SCS063_A_eye_tracking_log_60.0fps_20260702_140325.csv",
        "trial_summary": "SCS063_A_trial_summary_20260702_140325.csv",
        "stim_duration_ms": 800,
    },
}

def load_human_labels(subject_sess, col=0):
    """Load rater labels for a subject/session. Returns list with 1=look, 2=away."""
    r1_matches = glob.glob(os.path.join(rater1_dir, f"{subject_sess}*"))
    if not r1_matches:
        raise FileNotFoundError(f"No rater 1 file for {subject_sess}")
    df = pd.read_excel(r1_matches[0], header=None)
    labels = df.iloc[:, col].replace(0, 2).tolist()
    return [int(l) for l in labels]

def load_human_labels_r2(subject_sess, col=0):
    r2_matches = glob.glob(os.path.join(rater2_dir, f"{subject_sess}*"))
    if not r2_matches:
        return None
    df = pd.read_excel(r2_matches[0], header=None)
    labels = df.iloc[:, col].replace(0, 2).tolist()
    return [int(l) for l in labels]

def extract_trial_features(tracking_log_path, trial_summary_path, stim_duration_ms=800):
    """Extract per-trial feature vectors from the frame-level tracking log."""
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
    
    ts = pd.read_csv(trial_summary_path)
    ts.columns = ts.columns.str.strip()
    
    fps = 60.0
    frame_dur_ms = 1000.0 / fps
    
    feature_rows = []
    trial_ids = []
    
    for _, trial_row in ts.iterrows():
        tid = int(trial_row['trial_id'])
        stim_start = float(trial_row['start_time_ms'])
        stim_end = stim_start + stim_duration_ms
        
        # Select frames in stimulus window based on Trial ID column
        mask = df['trial_id'] == tid
        window = df[mask]
        
        # Fallback: if no frames tagged with trial_id, use frame number range
        if len(window) == 0:
            approx_start_frame = int(stim_start / frame_dur_ms)
            approx_end_frame = int(stim_end / frame_dur_ms)
            window = df[(df['frame_nr'] >= approx_start_frame) & (df['frame_nr'] < approx_end_frame)]
        
        n_frames = len(window)
        
        if n_frames == 0:
            feat = [np.nan] * 33
        else:
            has_face = ((window['l_cx'] != 0) | (window['r_cx'] != 0)).values.astype(float)
            n_face = has_face.sum()
            face_frac = n_face / n_frames
            
            w = window[(window['l_cx'] != 0) | (window['r_cx'] != 0)]
            
            def stats(arr):
                if len(arr) == 0:
                    return [np.nan, np.nan, np.nan, np.nan]
                arr = np.array(arr, dtype=np.float32)
                return [np.mean(arr), np.std(arr),
                        np.percentile(arr, 10), np.percentile(arr, 90)]
            
            pitch_stats = stats(w['pitch'].values)
            yaw_stats = stats(w['yaw'].values)
            l_dx_stats = stats(w['l_dx'].values)
            l_dy_stats = stats(w['l_dy'].values)
            r_dx_stats = stats(w['r_dx'].values)
            r_dy_stats = stats(w['r_dy'].values)
            
            if len(w) > 0:
                eye_sum = (w['l_dx'] + w['r_dx']).values
                eye_sum_stats = stats(eye_sum)
                head_centered_frac = float(
                    ((np.abs(w['pitch'].values) < 20) & (np.abs(w['yaw'].values) < 20)).mean()
                )
                eye_centered_frac = float((np.abs(eye_sum) < 5).mean())
            else:
                eye_sum_stats = [np.nan] * 4
                head_centered_frac = np.nan
                eye_centered_frac = np.nan
            
            feat = (
                [n_face, face_frac, n_frames]
                + pitch_stats
                + yaw_stats
                + l_dx_stats
                + l_dy_stats
                + r_dx_stats
                + r_dy_stats
                + eye_sum_stats
                + [head_centered_frac, eye_centered_frac]
            )
        
        feature_rows.append(feat)
        trial_ids.append(tid)
    
    X = np.array(feature_rows, dtype=np.float32)
    for col_i in range(X.shape[1]):
        col_vals = X[:, col_i]
        nan_mask = np.isnan(col_vals)
        if nan_mask.all():
            X[:, col_i] = 0.0
        elif nan_mask.any():
            X[nan_mask, col_i] = np.nanmean(col_vals)
    
    return X, trial_ids


def run_loocv():
    print("=" * 70)
    print("  OCAPI ML CLASSIFIER — Leave-One-Subject-Out Cross-Validation")
    print("=" * 70)
    
    all_X = {}
    all_y_r1 = {}
    all_y_r2 = {}
    
    for subj, cfg in SUBJECTS.items():
        print(f"\nLoading {subj}...")
        tracking_path = os.path.join(log_dir, cfg["tracking_log"])
        trial_path = os.path.join(log_dir, cfg["trial_summary"])
        
        X, trial_ids = extract_trial_features(tracking_path, trial_path, cfg["stim_duration_ms"])
        y_r1 = load_human_labels(subj)
        y_r2 = load_human_labels_r2(subj)
        
        n = min(len(X), len(y_r1))
        X = X[:n]
        y_r1 = y_r1[:n]
        if y_r2:
            y_r2 = y_r2[:min(len(y_r2), n)]
        
        all_X[subj] = X
        all_y_r1[subj] = np.array(y_r1)
        all_y_r2[subj] = np.array(y_r2) if y_r2 else None
        
        print(f"  {len(X)} trials, {X.shape[1]} features")
    
    subjects = list(SUBJECTS.keys())
    results = []
    
    feat_names = (
        ['n_face', 'face_frac', 'n_frames']
        + [f'pitch_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + [f'yaw_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + [f'l_dx_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + [f'l_dy_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + [f'r_dx_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + [f'r_dy_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + [f'eye_sum_{s}' for s in ['mean', 'std', 'p10', 'p90']]
        + ['head_centered_frac', 'eye_centered_frac']
    )
    
    print("\n" + "-" * 70)
    print("Running Leave-One-Subject-Out CV...")
    print("-" * 70)
    
    for test_subj in subjects:
        train_subjects = [s for s in subjects if s != test_subj]
        
        X_train = np.vstack([all_X[s] for s in train_subjects])
        y_train = np.concatenate([all_y_r1[s] for s in train_subjects])
        
        X_test = all_X[test_subj]
        y_test_r1 = all_y_r1[test_subj]
        y_test_r2 = all_y_r2[test_subj]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        clf.fit(X_train_s, y_train)
        
        y_pred = clf.predict(X_test_s)
        
        n = min(len(y_pred), len(y_test_r1))
        kappa_r1 = cohen_kappa_score(y_test_r1[:n], y_pred[:n], labels=[1, 2])
        agree_r1 = float(np.mean(y_test_r1[:n] == y_pred[:n]))
        
        kappa_r2 = None
        agree_r2 = None
        if y_test_r2 is not None:
            n2 = min(len(y_pred), len(y_test_r2))
            kappa_r2 = cohen_kappa_score(y_test_r2[:n2], y_pred[:n2], labels=[1, 2])
            agree_r2 = float(np.mean(y_test_r2[:n2] == y_pred[:n2]))
        
        mean_kappa = (kappa_r1 + kappa_r2) / 2.0 if kappa_r2 is not None else kappa_r1
        
        print(f"\n  Test subject: {test_subj} ({n} trials)")
        print(f"  Training on: {', '.join(train_subjects)}")
        print(f"  -> Rater 1 Kappa: {kappa_r1:.4f}  (Agreement: {agree_r1:.4f})")
        if kappa_r2 is not None:
            print(f"  -> Rater 2 Kappa: {kappa_r2:.4f}  (Agreement: {agree_r2:.4f})")
        print(f"  -> Mean Kappa:    {mean_kappa:.4f}")
        
        importances = clf.feature_importances_
        top5 = np.argsort(importances)[::-1][:5]
        print(f"  Top-5 features: {', '.join(f'{feat_names[i]}={importances[i]:.3f}' for i in top5)}")
        
        results.append({
            'subject': test_subj,
            'trials': n,
            'kappa_r1': round(kappa_r1, 4),
            'agree_r1': round(agree_r1, 4),
            'kappa_r2': round(kappa_r2, 4) if kappa_r2 is not None else 'N/A',
            'agree_r2': round(agree_r2, 4) if agree_r2 is not None else 'N/A',
            'mean_kappa': round(mean_kappa, 4),
        })
    
    df_results = pd.DataFrame(results)
    overall = df_results['mean_kappa'].mean()
    
    print("\n" + "=" * 70)
    print("  LOOCV SUMMARY TABLE")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print(f"\nOverall Average Cohen's Kappa (LOOCV): {overall:.4f}")
    print("=" * 70)
    
    out_path = os.path.join(project_root, "ml_classifier_results.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    
    return overall


if __name__ == "__main__":
    run_loocv()
