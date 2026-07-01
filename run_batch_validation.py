#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd
from datetime import datetime

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocapi.ocapi import Ocapi
from inter_rater_reliability import calculate_cohens_kappa

def run_validation():
    print("=" * 80)
    print("               OCAPI BATCH VALIDATION PIPELINE")
    print("=" * 80)
    
    # Path configuration
    video_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/all_videos_combined"
    eeg_base_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/input"
    rater_1_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_1"
    rater_2_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_2"
    output_log_dir = os.path.join(project_root, "validation_logs")
    os.makedirs(output_log_dir, exist_ok=True)
    
    video_extensions = ['.mkv', '.mp4', '.avi', '.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
    
    video_files = sorted(video_files)
    print(f"Found {len(video_files)} video files to process in {video_dir}.")
    
    results = []
    
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n{'-'*40}")
        print(f"Processing Subject/Session from video: {os.path.basename(video_path)}")
        
        # Parse subject and session
        try:
            parts = base_name.split('_')
            subject_id = parts[0]
            session = parts[1]
        except Exception as e:
            print(f"  Warning: Could not parse subject/session from filename. Skipping. ({e})")
            continue
            
        subject_sess = f"{subject_id}_{session}"
        
        # 1. Find corresponding EEG files for synchronization
        eeg_path = os.path.join(eeg_base_dir, subject_id, session)
        try:
            found_headers = glob.glob(os.path.join(eeg_path, f"{subject_sess}*.vhdr"))
            if not found_headers:
                raise FileNotFoundError(f"EEG header file not found in {eeg_path}")
            eeg_header_file = found_headers[0]
            
            found_markers = glob.glob(os.path.join(eeg_path, f"{subject_sess}*.vmrk"))
            if not found_markers:
                raise FileNotFoundError(f"EEG marker file not found in {eeg_path}")
            eeg_marker_file = found_markers[0]
        except Exception as e:
            print(f"  Skipping: EEG files missing or error: {e}")
            continue
            
        # 2. Find corresponding human coding files
        rater1_file = os.path.join(rater_1_dir, f"{subject_sess}_VideoCoding.xlsx")
        rater2_file = os.path.join(rater_2_dir, f"{subject_sess}_VideoCoding.xlsx")
        
        if not os.path.exists(rater1_file):
            # Try matching with glob if exact match not found
            glob_matches = glob.glob(os.path.join(rater_1_dir, f"{subject_sess}*"))
            if glob_matches:
                rater1_file = glob_matches[0]
            else:
                print(f"  Skipping: Ground truth coding file not found for {subject_sess} in {rater_1_dir}")
                continue
                
        # 3. Initialize Ocapi and run
        print(f"  Initializing Ocapi tracker...")
        try:
            tracker = Ocapi(
                subject_id=subject_id,
                session=session,
                config_file_path="config.yml",
                WEBCAM=None,
                VIDEO_INPUT=video_path,
                VIDEO_OUTPUT=None, # Disable output video encoding for speed
                TRACKING_DATA_LOG_FOLDER=output_log_dir
            )
            # Ensure visual elements are disabled
            tracker.SHOW_ON_SCREEN_DATA = False
            
            # EEG synchronization
            print(f"  Synchronizing with EEG data...")
            tracker.sync_with_eeg_and_set_onsets(
                header_file=eeg_header_file,
                marker_file=eeg_marker_file,
                stimulus_description="Stimulus",
                events_of_interest=["S 21", "S 22", "S 23", "S 24"]
            )
            
            print(f"  Running tracking analysis...")
            run_start_time = datetime.now()
            tracker.run()
            
            # Find the generated trial summary file
            # Sleep briefly to ensure file handle is released
            import time
            time.sleep(1)
            
            summary_pattern = os.path.join(output_log_dir, f"{subject_id}_trial_summary*.csv")
            generated_files = glob.glob(summary_pattern)
            if not generated_files:
                raise FileNotFoundError(f"No trial summary files generated matching pattern {summary_pattern}")
                
            # Filter files created during this run
            run_files = [f for f in generated_files if os.path.getmtime(f) > run_start_time.timestamp() - 5]
            if not run_files:
                raise FileNotFoundError(f"Could not find the trial summary file created during this run.")
            generated_summary_file = max(run_files, key=os.path.getmtime)
            
            # 4. Calculate Cohen's Kappa
            print(f"  Calculating Cohen's Kappa against Rater 1: {os.path.basename(rater1_file)}")
            kappa_1, summary_1 = calculate_cohens_kappa(
                file_path1=rater1_file,
                file_path2=generated_summary_file,
                column_index1=0,
                column_index2=6,
                labels=[1, 2]
            )
            
            kappa_2, agreement_2 = None, None
            if os.path.exists(rater2_file):
                print(f"  Calculating Cohen's Kappa against Rater 2: {os.path.basename(rater2_file)}")
                kappa_2, summary_2 = calculate_cohens_kappa(
                    file_path1=rater2_file,
                    file_path2=generated_summary_file,
                    column_index1=0,
                    column_index2=6,
                    labels=[1, 2]
                )
                if summary_2:
                    agreement_2 = summary_2.get("observed_agreement_proportion", 0)
            
            if kappa_1 is not None and summary_1 is not None:
                agreement_1 = summary_1.get("observed_agreement_proportion", 0)
                mean_kappa = (kappa_1 + kappa_2) / 2.0 if kappa_2 is not None else kappa_1
                
                print(f"  -> Rater 1 Kappa: {kappa_1:.4f} (Agreement: {agreement_1:.4f})")
                if kappa_2 is not None:
                    print(f"  -> Rater 2 Kappa: {kappa_2:.4f} (Agreement: {agreement_2:.4f})")
                print(f"  -> Mean Kappa: {mean_kappa:.4f}")
                
                results.append({
                    "subject": subject_sess,
                    "trials": summary_1.get("total_observations", 0),
                    "rater1_kappa": kappa_1,
                    "rater1_agreement": agreement_1,
                    "rater2_kappa": kappa_2 if kappa_2 is not None else "N/A",
                    "rater2_agreement": agreement_2 if agreement_2 is not None else "N/A",
                    "mean_kappa": mean_kappa
                })
            else:
                print(f"  -> Error: Could not calculate Kappa for {subject_sess}")
                
        except Exception as e:
            print(f"  Error processing {subject_sess}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("                       VALIDATION SUMMARY TABLE")
    print("=" * 80)
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # Save results to a CSV file
        summary_csv_path = os.path.join(project_root, "validation_run_summary.csv")
        df_results.to_csv(summary_csv_path, index=False)
        print(f"\nDetailed summary saved to: {summary_csv_path}")
        
        # Print overall average Kappa
        overall_mean_kappa = df_results["mean_kappa"].mean()
        print(f"\nOverall Average Cohen's Kappa across all processed subjects: {overall_mean_kappa:.4f}")
    else:
        print("No subjects were successfully validated.")
    print("=" * 80)

if __name__ == "__main__":
    run_validation()
