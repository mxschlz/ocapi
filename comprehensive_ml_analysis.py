#!/usr/bin/env python3
"""
comprehensive_ml_analysis.py

Comprehensive analysis for OCAPI ML gaze classifier:
1. Human-to-human Cohen's Kappa (rater 1 vs rater 2)
2. Multi-algorithm comparison (GradientBoosting, ExtraTrees, RandomForest)
3. Feature ablation: gaze-only vs head-pose-only vs combined vs extended
4. Leave-One-Subject-Out cross-validation across ALL available subjects
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
rater1_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_1"
rater2_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_2"
log_dir = os.path.join(project_root, "validation_logs")

# ── auto-discover subjects from validation_logs ────────────────────────────
def discover_subjects():
    """Find all subjects with both tracking log and trial summary in validation_logs."""
    import re
    subjects = {}
    # Find the most recent tracking log per subject
    tracking_files = sorted(glob.glob(os.path.join(log_dir, "*_eye_tracking_log_*.csv")))
    for f in tracking_files:
        basename = os.path.basename(f)
        # Extract subject like SCS048_A from SCS048_A_eye_tracking_log_60.0fps_20260702_135354.csv
        match = re.match(r"([A-Z]+\d+_[A-Z])_eye_tracking_log_", basename)
        if match:
            subj = match.group(1)
            subjects.setdefault(subj, {})["tracking_log"] = basename  # last one wins (sorted)
    
    # Find the most recent trial summary per subject (prefer _ml.csv if available)
    trial_files = sorted(glob.glob(os.path.join(log_dir, "*_trial_summary_*.csv")))
    for f in trial_files:
        basename = os.path.basename(f)
        match = re.match(r"([A-Z]+\d+_[A-Z])_trial_summary_", basename)
        if match:
            subj = match.group(1)
            subjects.setdefault(subj, {})["trial_summary"] = basename
    
    # Filter to only subjects with both files
    valid = {}
    for subj, cfg in subjects.items():
        if "tracking_log" in cfg and "trial_summary" in cfg:
            cfg["stim_duration_ms"] = 800
            valid[subj] = cfg
    return valid


# ── label loading ──────────────────────────────────────────────────────────
def load_human_labels(subject_sess, rater_dir, col=0):
    """Load rater labels. Returns list with 1=look, 2=away, or None if not found."""
    matches = glob.glob(os.path.join(rater_dir, f"{subject_sess}*"))
    if not matches:
        return None
    df = pd.read_excel(matches[0], header=None)
    labels = df.iloc[:, col].replace(0, 2).tolist()
    return [int(l) for l in labels if not pd.isna(l)]


# ── feature extraction ─────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "base": ["n_face", "face_frac", "n_frames"],
    "head_pose": [f"{a}_{s}" for a in ["pitch", "yaw"] for s in ["mean", "std", "p10", "p90"]],
    "eye_gaze": (
        [f"{e}_{s}" for e in ["l_dx", "l_dy", "r_dx", "r_dy"] for s in ["mean", "std", "p10", "p90"]]
        + [f"eye_sum_{s}" for s in ["mean", "std", "p10", "p90"]]
    ),
    "derived": ["head_centered_frac", "eye_centered_frac"],
    "roll": [f"roll_{s}" for s in ["mean", "std", "p10", "p90"]],
    "velocity": (
        [f"pitch_vel_{s}" for s in ["mean", "std", "p10", "p90"]]
        + [f"yaw_vel_{s}" for s in ["mean", "std", "p10", "p90"]]
        + [f"ldx_vel_{s}" for s in ["mean", "std", "p10", "p90"]]
        + [f"rdx_vel_{s}" for s in ["mean", "std", "p10", "p90"]]
    ),
    "vergence": [f"vergence_{s}" for s in ["mean", "std", "p10", "p90"]],
    "blink": ["blink_frac"],
}


def stats(arr):
    if len(arr) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    arr = np.array(arr, dtype=np.float64)
    return [np.nanmean(arr), np.nanstd(arr),
            np.nanpercentile(arr, 10), np.nanpercentile(arr, 90)]


def extract_extended_trial_features(tracking_log_path, trial_summary_path, stim_duration_ms=800):
    """Extract per-trial feature vectors with extended features (including roll, velocity, vergence, blinks)."""
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
    
    # Check if head pose columns exist
    has_pitch = 'pitch' in df.columns
    has_yaw = 'yaw' in df.columns
    has_roll = 'roll' in df.columns
    has_blinks = 'blinks' in df.columns
    
    ts = pd.read_csv(trial_summary_path)
    ts.columns = ts.columns.str.strip()
    
    fps = 60.0
    frame_dur_ms = 1000.0 / fps
    
    feature_rows = []
    trial_ids = []
    
    # Total number of features:
    # base(3) + head_pose(8) + eye_gaze(20) + derived(2) + roll(4) + velocity(16) + vergence(4) + blink(1) = 58
    N_FEATURES = 58
    
    for _, trial_row in ts.iterrows():
        tid = int(trial_row['trial_id'])
        stim_start = float(trial_row['start_time_ms'])
        stim_end = stim_start + stim_duration_ms
        
        mask = df['trial_id'] == tid
        window = df[mask]
        
        if len(window) == 0:
            approx_start_frame = int(stim_start / frame_dur_ms)
            approx_end_frame = int(stim_end / frame_dur_ms)
            window = df[(df['frame_nr'] >= approx_start_frame) & (df['frame_nr'] < approx_end_frame)]
        
        n_frames = len(window)
        
        if n_frames == 0:
            feat = [np.nan] * N_FEATURES
        else:
            has_face = ((window['l_cx'] != 0) | (window['r_cx'] != 0)).values.astype(float)
            n_face = has_face.sum()
            face_frac = n_face / n_frames
            
            w = window[(window['l_cx'] != 0) | (window['r_cx'] != 0)]
            
            # Base features
            base_feat = [n_face, face_frac, n_frames]
            
            # Head pose features
            if has_pitch and len(w) > 0:
                pitch_stats = stats(w['pitch'].values)
                yaw_stats = stats(w['yaw'].values)
            else:
                pitch_stats = [np.nan] * 4
                yaw_stats = [np.nan] * 4
            head_pose_feat = pitch_stats + yaw_stats
            
            # Eye gaze features
            if len(w) > 0:
                l_dx_stats = stats(w['l_dx'].values)
                l_dy_stats = stats(w['l_dy'].values)
                r_dx_stats = stats(w['r_dx'].values)
                r_dy_stats = stats(w['r_dy'].values)
                eye_sum = (w['l_dx'] + w['r_dx']).values
                eye_sum_stats = stats(eye_sum)
            else:
                l_dx_stats = l_dy_stats = r_dx_stats = r_dy_stats = [np.nan] * 4
                eye_sum_stats = [np.nan] * 4
                eye_sum = np.array([])
            eye_gaze_feat = l_dx_stats + l_dy_stats + r_dx_stats + r_dy_stats + eye_sum_stats
            
            # Derived fractions
            if len(w) > 0 and has_pitch:
                head_centered_frac = float(
                    ((np.abs(w['pitch'].values) < 20) & (np.abs(w['yaw'].values) < 20)).mean()
                )
            else:
                head_centered_frac = np.nan
            if len(eye_sum) > 0:
                eye_centered_frac = float((np.abs(eye_sum) < 5).mean())
            else:
                eye_centered_frac = np.nan
            derived_feat = [head_centered_frac, eye_centered_frac]
            
            # Roll features (NEW)
            if has_roll and len(w) > 0:
                roll_feat = stats(w['roll'].values)
            else:
                roll_feat = [np.nan] * 4
            
            # Velocity features (NEW): frame-to-frame changes
            if len(w) > 1:
                if has_pitch:
                    pitch_vel_stats = stats(np.diff(w['pitch'].values))
                    yaw_vel_stats = stats(np.diff(w['yaw'].values))
                else:
                    pitch_vel_stats = [np.nan] * 4
                    yaw_vel_stats = [np.nan] * 4
                ldx_vel_stats = stats(np.diff(w['l_dx'].values))
                rdx_vel_stats = stats(np.diff(w['r_dx'].values))
            else:
                pitch_vel_stats = yaw_vel_stats = ldx_vel_stats = rdx_vel_stats = [np.nan] * 4
            velocity_feat = pitch_vel_stats + yaw_vel_stats + ldx_vel_stats + rdx_vel_stats
            
            # Vergence features (NEW): difference between L and R horizontal gaze
            if len(w) > 0:
                vergence = (w['l_dx'] - w['r_dx']).values
                vergence_feat = stats(vergence)
            else:
                vergence_feat = [np.nan] * 4
            
            # Blink features (NEW)
            if has_blinks and n_frames > 0:
                no_face_frac = 1.0 - face_frac
                blink_feat = [no_face_frac]
            else:
                blink_feat = [np.nan]
            
            feat = base_feat + head_pose_feat + eye_gaze_feat + derived_feat + roll_feat + velocity_feat + vergence_feat + blink_feat
        
        feature_rows.append(feat)
        trial_ids.append(tid)
    
    X = np.array(feature_rows, dtype=np.float64)
    
    # Impute NaN with column mean
    for col_i in range(X.shape[1]):
        col_vals = X[:, col_i]
        nan_mask = np.isnan(col_vals)
        if nan_mask.all():
            X[:, col_i] = 0.0
        elif nan_mask.any():
            X[nan_mask, col_i] = np.nanmean(col_vals)
    
    return X, trial_ids


# ── Feature set definitions ───────────────────────────────────────────────
# Indices into the 58-feature vector:
# [0:3] base | [3:11] head_pose | [11:31] eye_gaze | [31:33] derived
# [33:37] roll | [37:53] velocity | [53:57] vergence | [57] blink
FEATURE_SETS = {
    "gaze_only": list(range(0, 3)) + list(range(11, 33)),          # base + eye_gaze + derived (no head)
    "head_only": list(range(0, 11)) + [31],                         # base + head_pose + head_centered_frac
    "combined_current": list(range(0, 33)),                          # base + head_pose + eye_gaze + derived (matches current 33-feat)
    "extended_all": list(range(0, 58)),                              # everything including roll, velocity, vergence, blink
    "gaze+velocity": list(range(0, 3)) + list(range(11, 33)) + list(range(41, 53)) + list(range(53, 57)) + [57],  # gaze + eye velocity + vergence + blink (no head)
    "combined+velocity": list(range(0, 33)) + list(range(33, 58)),   # current + all new features (same as extended_all)
}

ALL_FEATURE_NAMES = (
    FEATURE_GROUPS["base"]
    + FEATURE_GROUPS["head_pose"]
    + FEATURE_GROUPS["eye_gaze"]
    + FEATURE_GROUPS["derived"]
    + FEATURE_GROUPS["roll"]
    + FEATURE_GROUPS["velocity"]
    + FEATURE_GROUPS["vergence"]
    + FEATURE_GROUPS["blink"]
)

# ── Algorithms ─────────────────────────────────────────────────────────────
ALGORITHMS = {
    "GradientBoosting": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42
    ),
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
}


# ── Main analysis ──────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  OCAPI COMPREHENSIVE ML ANALYSIS")
    print("  Human-to-Human Kappa + Algorithm × Feature Set Comparison")
    print("=" * 80)
    
    # ── 1. Discover subjects ──
    subjects_config = discover_subjects()
    print(f"\nDiscovered {len(subjects_config)} subjects with tracking data:")
    for subj in sorted(subjects_config.keys()):
        print(f"  {subj}")
    
    # ── 2. Load all data ──
    all_X = {}
    all_y_r1 = {}
    all_y_r2 = {}
    subject_list = []
    
    for subj in sorted(subjects_config.keys()):
        cfg = subjects_config[subj]
        tracking_path = os.path.join(log_dir, cfg["tracking_log"])
        trial_path = os.path.join(log_dir, cfg["trial_summary"])
        
        y_r1 = load_human_labels(subj, rater1_dir)
        if y_r1 is None:
            print(f"  Skipping {subj}: no rater 1 labels")
            continue
        
        y_r2 = load_human_labels(subj, rater2_dir)
        
        print(f"\nLoading {subj}...")
        X, trial_ids = extract_extended_trial_features(tracking_path, trial_path, cfg["stim_duration_ms"])
        
        n = min(len(X), len(y_r1))
        X = X[:n]
        y_r1 = y_r1[:n]
        if y_r2:
            y_r2 = y_r2[:min(len(y_r2), n)]
            # Pad with NaN if y_r2 is shorter
            if len(y_r2) < n:
                y_r2 = y_r2 + [np.nan] * (n - len(y_r2))
        
        all_X[subj] = X
        all_y_r1[subj] = np.array(y_r1)
        all_y_r2[subj] = np.array(y_r2) if y_r2 else None
        subject_list.append(subj)
        
        print(f"  {n} trials, {X.shape[1]} features, rater2={'yes' if y_r2 else 'no'}")
    
    # ── 3. Human-to-human Cohen's Kappa ──
    print("\n" + "=" * 80)
    print("  SECTION 1: HUMAN-TO-HUMAN INTER-RATER RELIABILITY")
    print("=" * 80)
    
    h2h_results = []
    for subj in subject_list:
        y_r1 = all_y_r1[subj]
        y_r2 = all_y_r2.get(subj)
        if y_r2 is None:
            continue
        
        # Find valid (non-NaN) indices for both raters
        valid = ~np.isnan(y_r1.astype(float)) & ~np.isnan(y_r2.astype(float))
        if valid.sum() == 0:
            continue
        
        r1_valid = y_r1[valid].astype(int)
        r2_valid = y_r2[valid].astype(int)
        
        # Harmonize: convert 0 -> 2
        r1_valid[r1_valid == 0] = 2
        r2_valid[r2_valid == 0] = 2
        
        kappa = cohen_kappa_score(r1_valid, r2_valid, labels=[1, 2])
        agree = accuracy_score(r1_valid, r2_valid)
        cm = confusion_matrix(r1_valid, r2_valid, labels=[1, 2])
        
        h2h_results.append({
            'subject': subj,
            'trials': int(valid.sum()),
            'h2h_kappa': round(kappa, 4),
            'h2h_agreement': round(agree, 4),
            'both_look': int(cm[0, 0]),
            'both_away': int(cm[1, 1]),
            'r1_look_r2_away': int(cm[0, 1]),
            'r1_away_r2_look': int(cm[1, 0]),
        })
        
        print(f"  {subj:12s}: κ = {kappa:+.4f}  agreement = {agree:.4f}  "
              f"(n={valid.sum()}, look-look={cm[0,0]}, away-away={cm[1,1]})")
    
    h2h_df = pd.DataFrame(h2h_results)
    if len(h2h_df) > 0:
        overall_h2h = h2h_df['h2h_kappa'].mean()
        median_h2h = h2h_df['h2h_kappa'].median()
        print(f"\n  ─── Human-to-Human Summary ───")
        print(f"  Subjects with both raters: {len(h2h_df)}")
        print(f"  Mean κ:   {overall_h2h:.4f}")
        print(f"  Median κ: {median_h2h:.4f}")
        print(f"  Range:    [{h2h_df['h2h_kappa'].min():.4f}, {h2h_df['h2h_kappa'].max():.4f}]")
        
        h2h_df.to_csv(os.path.join(project_root, "human_to_human_kappa.csv"), index=False)
        print(f"\n  Saved to: human_to_human_kappa.csv")
    
    # ── 4. ML Algorithm × Feature Set LOSO CV ──
    print("\n" + "=" * 80)
    print("  SECTION 2: ML ALGORITHM × FEATURE SET COMPARISON (LOSO)")
    print("=" * 80)
    
    all_results = []
    
    for algo_name, algo_factory in ALGORITHMS.items():
        for fset_name, fset_indices in FEATURE_SETS.items():
            print(f"\n  ── {algo_name} × {fset_name} ({len(fset_indices)} features) ──")
            
            fold_results = []
            all_y_true = []
            all_y_pred = []
            
            for test_subj in subject_list:
                train_subjects = [s for s in subject_list if s != test_subj]
                
                if len(train_subjects) == 0:
                    continue
                
                X_train = np.vstack([all_X[s][:, fset_indices] for s in train_subjects])
                y_train = np.concatenate([all_y_r1[s] for s in train_subjects])
                
                X_test = all_X[test_subj][:, fset_indices]
                y_test_r1 = all_y_r1[test_subj]
                y_test_r2 = all_y_r2[test_subj]
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                clf = algo_factory()
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                
                n = min(len(y_pred), len(y_test_r1))
                kappa_r1 = cohen_kappa_score(y_test_r1[:n], y_pred[:n], labels=[1, 2])
                agree_r1 = accuracy_score(y_test_r1[:n], y_pred[:n])
                
                kappa_r2 = np.nan
                if y_test_r2 is not None:
                    valid_r2 = ~np.isnan(y_test_r2[:n].astype(float))
                    if valid_r2.sum() > 0:
                        kappa_r2 = cohen_kappa_score(
                            y_test_r2[:n][valid_r2].astype(int),
                            y_pred[:n][valid_r2],
                            labels=[1, 2]
                        )
                
                mean_kappa = kappa_r1 if np.isnan(kappa_r2) else (kappa_r1 + kappa_r2) / 2
                
                fold_results.append({
                    'subject': test_subj,
                    'kappa_r1': kappa_r1,
                    'kappa_r2': kappa_r2,
                    'mean_kappa': mean_kappa,
                    'agree_r1': agree_r1,
                })
                
                all_y_true.extend(y_test_r1[:n].tolist())
                all_y_pred.extend(y_pred[:n].tolist())
            
            fold_df = pd.DataFrame(fold_results)
            avg_kappa_r1 = fold_df['kappa_r1'].mean()
            avg_kappa_r2 = fold_df['kappa_r2'].dropna().mean() if fold_df['kappa_r2'].notna().any() else np.nan
            avg_mean_kappa = fold_df['mean_kappa'].mean()
            avg_agree = fold_df['agree_r1'].mean()
            
            # Global kappa (pooled across all folds)
            global_kappa = cohen_kappa_score(all_y_true, all_y_pred, labels=[1, 2])
            
            print(f"    Avg Kappa (R1): {avg_kappa_r1:.4f}  |  Avg Kappa (R2): {avg_kappa_r2:.4f}  |  "
                  f"Avg Mean κ: {avg_mean_kappa:.4f}  |  Global κ: {global_kappa:.4f}  |  Agreement: {avg_agree:.4f}")
            
            # Feature importance (train on all data for one summary)
            X_all = np.vstack([all_X[s][:, fset_indices] for s in subject_list])
            y_all = np.concatenate([all_y_r1[s] for s in subject_list])
            scaler_all = StandardScaler()
            X_all_s = scaler_all.fit_transform(X_all)
            clf_all = algo_factory()
            clf_all.fit(X_all_s, y_all)
            
            if hasattr(clf_all, 'feature_importances_'):
                importances = clf_all.feature_importances_
                fnames = [ALL_FEATURE_NAMES[i] for i in fset_indices]
                top5_idx = np.argsort(importances)[::-1][:5]
                top5_str = ", ".join(f"{fnames[i]}={importances[i]:.3f}" for i in top5_idx)
                print(f"    Top-5 features: {top5_str}")
            
            all_results.append({
                'algorithm': algo_name,
                'feature_set': fset_name,
                'n_features': len(fset_indices),
                'n_subjects': len(subject_list),
                'avg_kappa_r1': round(avg_kappa_r1, 4),
                'avg_kappa_r2': round(avg_kappa_r2, 4) if not np.isnan(avg_kappa_r2) else 'N/A',
                'avg_mean_kappa': round(avg_mean_kappa, 4),
                'global_kappa': round(global_kappa, 4),
                'avg_agreement': round(avg_agree, 4),
            })
            
            # Save per-subject results for each config
            fold_df['algorithm'] = algo_name
            fold_df['feature_set'] = fset_name
            fold_df.to_csv(
                os.path.join(project_root, f"loso_detail_{algo_name}_{fset_name}.csv"),
                index=False
            )
    
    # ── 5. Summary table ──
    print("\n" + "=" * 80)
    print("  SUMMARY: ALGORITHM × FEATURE SET COMPARISON")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('avg_mean_kappa', ascending=False)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(project_root, "ml_algorithm_feature_comparison.csv"), index=False)
    print(f"\nSaved to: ml_algorithm_feature_comparison.csv")
    
    # ── 6. Best config comparison to human-to-human ──
    if len(h2h_df) > 0:
        print("\n" + "=" * 80)
        print("  CONTEXT: BEST ML vs HUMAN-TO-HUMAN RELIABILITY")
        print("=" * 80)
        best = results_df.iloc[0]
        print(f"  Human-to-Human mean κ:  {overall_h2h:.4f}  (across {len(h2h_df)} subjects)")
        print(f"  Best ML config:         {best['algorithm']} × {best['feature_set']}")
        print(f"    Average mean κ:       {best['avg_mean_kappa']}")
        print(f"    Global κ (pooled):    {best['global_kappa']}")
        print(f"    Agreement:            {best['avg_agreement']}")
        gap = overall_h2h - float(best['avg_mean_kappa'])
        print(f"  Gap (H2H - ML):         {gap:+.4f}")
    
    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results_df, h2h_df


if __name__ == "__main__":
    results_df, h2h_df = main()
