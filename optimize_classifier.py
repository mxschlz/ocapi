#!/usr/bin/env python3
"""
optimize_classifier.py

Optimizes the ExtraTrees Classifier for the OCAPI gaze classifier:
- Systematic Grid/Random Search over:
  - Latency offsets
  - Feature subsets
  - ExtraTrees hyperparameters
  - Decision probability thresholds
  - Oversampling strategy
  - Training target (Consensus vs Rater 1)
- Validation via Leave-One-Subject-Group-Out CV (to avoid leakage within subjects across sessions).
- Optimizes for both Gwet's AC1 and Cohen's Kappa against human raters.
"""

import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score
import random

project_root = os.path.dirname(os.path.abspath(__file__))
rater1_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_1"
rater2_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_2"
log_dir = os.path.join(project_root, "validation_logs")

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
        
    return np.array(feature_rows, dtype=np.float32)

def random_oversample(X, y):
    np.random.seed(42)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) <= 1:
        return X, y
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

def compute_gwet_ac1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    po = np.mean(y_true == y_pred)
    p1_true = np.mean(y_true == 1)
    p1_pred = np.mean(y_pred == 1)
    
    pi1 = (p1_true + p1_pred) / 2.0
    pe = 2.0 * pi1 * (1.0 - pi1)
    
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)

# Feature subsets in 54 layout
# [0:3] base | [3:11] head_pose | [11:31] eye_gaze | [31:47] diffs | [47:51] vergence | [51:54] derived+blink
FEATURE_SETS = {
    "gaze_only": [0, 1, 2] + list(range(11, 31)) + [52], # 24 features (existing)
    "gaze_diffs": [0, 1, 2] + list(range(11, 31)) + list(range(39, 47)) + [52], # 32 features
    "gaze_vergence": [0, 1, 2] + list(range(11, 31)) + list(range(47, 51)) + [52], # 28 features
    "gaze_blinks": [0, 1, 2] + list(range(11, 31)) + [52, 53], # 25 features
    "combined_gaze": [0, 1, 2] + list(range(11, 31)) + list(range(39, 51)) + [52, 53], # 38 features (gaze + gaze diffs + vergence + blink)
    "all_features": list(range(54)), # all 54 features (gaze + head pose)
}

def group_by_subject(subject_sess_list):
    """Group SCS048_A, SCS048_B into 'SCS048'"""
    groups = {}
    for subj_sess in subject_sess_list:
        subject_id = subj_sess.split('_')[0]
        groups.setdefault(subject_id, []).append(subj_sess)
    return groups

def main():
    print("=" * 80)
    print("  OCAPI EXTRA TREES CLASSIFIER OPTIMIZATION")
    print("=" * 80)
    
    subjects_config = discover_subjects()
    subject_list = sorted(list(subjects_config.keys()))
    print(f"Found {len(subject_list)} subject sessions in validation_logs.")
    if len(subject_list) == 0:
        print("Error: No subject sessions found. Run validation first.")
        return
        
    # Load all labels
    all_y_r1 = {}
    all_y_r2 = {}
    
    for subj in subject_list:
        y_r1 = load_human_labels(subj, rater1_dir)
        y_r2 = load_human_labels(subj, rater2_dir)
        all_y_r1[subj] = np.array(y_r1) if y_r1 else None
        all_y_r2[subj] = np.array(y_r2) if y_r2 else None
        
    # We filter subjects to only those that have rater 1 coding (since rater 1 is our main target)
    valid_subjects = [s for s in subject_list if all_y_r1[s] is not None]
    print(f"Subjects with Rater 1 coding: {len(valid_subjects)}")
    
    # Pre-extract raw features for different latency offsets to speed up hyperparameter search
    latency_offsets = [0, 50, 100, 150, 200, 250]
    raw_features = {}
    
    print("\nPre-extracting features for different latency offsets...")
    for offset in latency_offsets:
        print(f"  Extraction for latency_offset = {offset} ms...")
        raw_features[offset] = {}
        for subj in valid_subjects:
            cfg = subjects_config[subj]
            raw_X = extract_production_features(
                os.path.join(log_dir, cfg["tracking_log"]),
                os.path.join(log_dir, cfg["trial_summary"]),
                latency_offset_ms=offset,
                stim_duration_ms=cfg["stim_duration_ms"]
            )
            n = min(len(raw_X), len(all_y_r1[subj]))
            raw_features[offset][subj] = raw_X[:n]
            # align labels
            all_y_r1[subj] = all_y_r1[subj][:n]
            if all_y_r2[subj] is not None:
                all_y_r2[subj] = all_y_r2[subj][:n]
                
    # Normalize features per-subject (preprocessed)
    norm_features = {}
    for offset in latency_offsets:
        norm_features[offset] = robust_subject_normalize(raw_features[offset])
        
    # Group subjects for Group-K-Fold (Leave-One-Subject-Group-Out)
    subject_groups = group_by_subject(valid_subjects)
    group_keys = sorted(list(subject_groups.keys()))
    print(f"\nGrouped into {len(group_keys)} unique subject groups:")
    for grp, sessions in subject_groups.items():
        print(f"  {grp}: {', '.join(sessions)}")
        
    # Hyperparameter space definition
    param_grid = []
    
    # We define a structured set of candidates to search
    # Let's perform a comprehensive random search of, say, 150 configurations
    random.seed(42)
    
    feature_sets_keys = list(FEATURE_SETS.keys())
    
    # ExtraTrees parameters
    n_estimators_opts = [100, 200, 300, 400]
    max_depth_opts = [6, 8, 10, 12, 15, None]
    min_samples_leaf_opts = [1, 2, 4, 6]
    max_features_opts = ['sqrt', 'log2', None]
    class_weight_opts = [None, 'balanced']
    oversample_opts = [True, False]
    target_opts = ['consensus', 'rater1']
    threshold_opts = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    # Generate 150 unique param configs
    seen_configs = set()
    while len(param_grid) < 150:
        cfg = (
            random.choice(latency_offsets),
            random.choice(feature_sets_keys),
            random.choice(n_estimators_opts),
            random.choice(max_depth_opts),
            random.choice(min_samples_leaf_opts),
            random.choice(max_features_opts),
            random.choice(class_weight_opts),
            random.choice(oversample_opts),
            random.choice(target_opts),
            random.choice(threshold_opts),
        )
        if cfg not in seen_configs:
            seen_configs.add(cfg)
            param_grid.append({
                'latency_offset_ms': cfg[0],
                'feature_set_name': cfg[1],
                'n_estimators': cfg[2],
                'max_depth': cfg[3],
                'min_samples_leaf': cfg[4],
                'max_features': cfg[5],
                'class_weight': cfg[6],
                'oversample': cfg[7],
                'training_target': cfg[8],
                'threshold': cfg[9],
            })
            
    print(f"\nCreated hyperparameter search space with {len(param_grid)} configurations.")
    
    best_score = -1.0
    best_config = None
    results = []
    
    # Loop over configurations
    for run_i, params in enumerate(param_grid):
        offset = params['latency_offset_ms']
        fset_name = params['feature_set_name']
        fset_indices = FEATURE_SETS[fset_name]
        
        # Cross-validation arrays
        all_preds = []
        all_trues_r1 = []
        all_trues_r2 = []
        
        kappas_r1 = []
        kappas_r2 = []
        ac1s_r1 = []
        ac1s_r2 = []
        
        # Leave-One-Subject-Group-Out CV
        for test_group in group_keys:
            # Test sessions
            test_sessions = subject_groups[test_group]
            # Train sessions
            train_sessions = [s for s in valid_subjects if s not in test_sessions]
            
            # Prepare test data
            X_test_list = []
            y_test_r1_list = []
            y_test_r2_list = []
            
            for s in test_sessions:
                X_s = norm_features[offset][s][:, fset_indices]
                X_test_list.append(X_s)
                y_test_r1_list.append(all_y_r1[s])
                if all_y_r2[s] is not None:
                    # Rater 2 labels might contain NaNs (unrated trials), we preserve them
                    y_test_r2_list.append(all_y_r2[s])
                else:
                    y_test_r2_list.append(np.array([np.nan]*len(all_y_r1[s])))
                    
            X_test = np.vstack(X_test_list)
            y_test_r1 = np.concatenate(y_test_r1_list)
            y_test_r2 = np.concatenate(y_test_r2_list)
            
            # Prepare training data
            X_train_list = []
            y_train_list = []
            
            for s in train_sessions:
                X_s = norm_features[offset][s][:, fset_indices]
                y_s_r1 = all_y_r1[s]
                y_s_r2 = all_y_r2[s]
                
                # Training target strategy
                if params['training_target'] == 'consensus' and y_s_r2 is not None:
                    # Consensus: only trials where rater 1 and rater 2 agree and are valid
                    valid = ~np.isnan(y_s_r1.astype(float)) & ~np.isnan(y_s_r2.astype(float))
                    agree = (y_s_r1 == y_s_r2) & valid
                    if agree.sum() > 0:
                        X_train_list.append(X_s[agree])
                        y_train_list.append(y_s_r1[agree])
                else:
                    # Rater 1 target
                    valid = ~np.isnan(y_s_r1.astype(float))
                    if valid.sum() > 0:
                        X_train_list.append(X_s[valid])
                        y_train_list.append(y_s_r1[valid])
                        
            if len(X_train_list) == 0:
                continue
                
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            
            # Oversampling strategy
            if params['oversample']:
                X_train, y_train = random_oversample(X_train, y_train)
                
            # Scale
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # Classifier fit
            clf = ExtraTreesClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                class_weight=params['class_weight'],
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train_s, y_train)
            
            # Predict
            probs = clf.predict_proba(X_test_s)[:, 0] # Prob of class 1
            preds = np.where(probs >= params['threshold'], 1, 2)
            
            # Compute subject-fold level metrics
            # Against Rater 1
            k_r1 = cohen_kappa_score(y_test_r1, preds, labels=[1, 2])
            a_r1 = compute_gwet_ac1(y_test_r1, preds)
            kappas_r1.append(k_r1)
            ac1s_r1.append(a_r1)
            
            # Against Rater 2 (only for non-NaN labels)
            valid_r2 = ~np.isnan(y_test_r2.astype(float))
            if valid_r2.sum() > 0:
                k_r2 = cohen_kappa_score(y_test_r2[valid_r2].astype(int), preds[valid_r2], labels=[1, 2])
                a_r2 = compute_gwet_ac1(y_test_r2[valid_r2].astype(int), preds[valid_r2])
                kappas_r2.append(k_r2)
                ac1s_r2.append(a_r2)
                
        # Average metrics across subject folds
        avg_k_r1 = np.nanmean(kappas_r1)
        avg_a_r1 = np.nanmean(ac1s_r1)
        avg_k_r2 = np.nanmean(kappas_r2) if kappas_r2 else np.nan
        avg_a_r2 = np.nanmean(ac1s_r2) if ac1s_r2 else np.nan
        
        mean_kappa = (avg_k_r1 + avg_k_r2) / 2.0 if not np.isnan(avg_k_r2) else avg_k_r1
        mean_ac1 = (avg_a_r1 + avg_a_r2) / 2.0 if not np.isnan(avg_a_r2) else avg_a_r1
        
        # Overall optimization score
        score = (mean_kappa + mean_ac1) / 2.0
        
        results.append({
            'params': params,
            'avg_kappa_r1': avg_k_r1,
            'avg_ac1_r1': avg_a_r1,
            'avg_kappa_r2': avg_k_r2,
            'avg_ac1_r2': avg_a_r2,
            'mean_kappa': mean_kappa,
            'mean_ac1': mean_ac1,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_config = params
            print(f"[{run_i+1}/{len(param_grid)}] NEW BEST! Score: {score:.4f} (Kappa: {mean_kappa:.4f}, AC1: {mean_ac1:.4f}) | Offset: {offset}ms | Features: {fset_name}")
        elif (run_i + 1) % 15 == 0:
            print(f"[{run_i+1}/{len(param_grid)}] Current Best Score: {best_score:.4f}")
            
    # Compile results into df
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    results_df.to_csv(os.path.join(project_root, "optimization_search_results.csv"), index=False)
    print(f"\nAll results saved to: {os.path.join(project_root, 'optimization_search_results.csv')}")
    
    print("\n" + "=" * 80)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best combined score: {best_score:.4f}")
    print("Best hyperparameters:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print("=" * 80)
    
    # ── Re-train final production model on best config ──
    print("\nTraining final production model using optimized configuration...")
    
    best_offset = best_config['latency_offset_ms']
    best_fset_name = best_config['feature_set_name']
    best_fset_indices = FEATURE_SETS[best_fset_name]
    
    # Extract raw features for all subjects using best offset
    raw_X_prod = {}
    for subj in valid_subjects:
        cfg = subjects_config[subj]
        raw_X_prod[subj] = extract_production_features(
            os.path.join(log_dir, cfg["tracking_log"]),
            os.path.join(log_dir, cfg["trial_summary"]),
            latency_offset_ms=best_offset,
            stim_duration_ms=cfg["stim_duration_ms"]
        )
        
    norm_X_prod = robust_subject_normalize(raw_X_prod)
    
    # Train set construction
    X_train_list = []
    y_train_list = []
    
    for subj in valid_subjects:
        X_s = norm_X_prod[subj][:, best_fset_indices]
        y_s_r1 = all_y_r1[subj]
        y_s_r2 = all_y_r2[subj]
        
        if best_config['training_target'] == 'consensus' and y_s_r2 is not None:
            valid = ~np.isnan(y_s_r1.astype(float)) & ~np.isnan(y_s_r2.astype(float))
            agree = (y_s_r1 == y_s_r2) & valid
            if agree.sum() > 0:
                X_train_list.append(X_s[agree])
                y_train_list.append(y_s_r1[agree])
        else:
            valid = ~np.isnan(y_s_r1.astype(float))
            if valid.sum() > 0:
                X_train_list.append(X_s[valid])
                y_train_list.append(y_s_r1[valid])
                
    X_train_final = np.vstack(X_train_list)
    y_train_final = np.concatenate(y_train_list)
    
    if best_config['oversample']:
        X_train_final, y_train_final = random_oversample(X_train_final, y_train_final)
        
    scaler_final = StandardScaler()
    X_train_final_s = scaler_final.fit_transform(X_train_final)
    
    clf_final = ExtraTreesClassifier(
        n_estimators=best_config['n_estimators'],
        max_depth=best_config['max_depth'],
        min_samples_leaf=best_config['min_samples_leaf'],
        max_features=best_config['max_features'],
        class_weight=best_config['class_weight'],
        random_state=42,
        n_jobs=-1
    )
    clf_final.fit(X_train_final_s, y_train_final)
    
    # Save the final production model structure
    model_data = {
        "classifier": clf_final,
        "scaler": scaler_final,
        "feature_indices": best_fset_indices,
        "threshold": best_config['threshold'],
        "latency_offset_ms": best_offset
    }
    
    model_dir = os.path.join(project_root, "ocapi", "models")
    model_path = os.path.join(model_dir, "gaze_classifier_gb.pkl")
    backup_path = os.path.join(model_dir, "gaze_classifier_gb_backup.pkl")
    
    if os.path.exists(model_path):
        print(f"Creating backup of existing model at {backup_path}...")
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(model_path, backup_path)
        
    print(f"Saving optimized production model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print("Optimized production model successfully trained and saved!")

if __name__ == "__main__":
    main()
