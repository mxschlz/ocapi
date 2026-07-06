#!/usr/bin/env python3
import os
import sys
import glob
import gc
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ocapi.ocapi import Ocapi
from inter_rater_reliability import calculate_cohens_kappa

def process_single_video_task(args):
    """
    Processes a single video: runs Ocapi tracker and calculates Cohen's Kappa.
    Runs inside a separate process.
    """
    idx, total_files, video_path, eeg_base_dir, rater_1_dir, rater_2_dir, output_log_dir, force_rerun = args
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    try:
        parts = base_name.split('_')
        subject_id = parts[0]
        session = parts[1]
    except Exception as e:
        return {"status": "error", "message": f"Could not parse subject/session from filename {base_name}: {e}"}
        
    subject_sess = f"{subject_id}_{session}"
    
    # 1. Find corresponding EEG files
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
        return {"status": "skipped", "message": f"EEG files missing or error: {e}"}
        
    # 2. Find corresponding human coding files
    rater1_file = os.path.join(rater_1_dir, f"{subject_sess}_VideoCoding.xlsx")
    rater2_file = os.path.join(rater_2_dir, f"{subject_sess}_VideoCoding.xlsx")
    if not os.path.exists(rater2_file):
        glob_matches = glob.glob(os.path.join(rater_2_dir, f"{subject_sess}*"))
        if glob_matches:
            rater2_file = glob_matches[0]
            
    if not os.path.exists(rater1_file):
        glob_matches = glob.glob(os.path.join(rater_1_dir, f"{subject_sess}*"))
        if glob_matches:
            rater1_file = glob_matches[0]
        else:
            return {"status": "skipped", "message": f"Ground truth coding file not found for {subject_sess} in {rater_1_dir}"}
            
    # 3. Check for existing summary file from today to support crash resumption
    summary_pattern = os.path.join(output_log_dir, f"{subject_sess}_trial_summary_20260702_*.csv")
    existing_summaries = glob.glob(summary_pattern)
    
    today_str = datetime.now().strftime("%Y%m%d")
    if not existing_summaries:
        summary_pattern = os.path.join(output_log_dir, f"{subject_sess}_trial_summary_{today_str}_*.csv")
        existing_summaries = glob.glob(summary_pattern)
        
    generated_summary_file = None
    if existing_summaries and not force_rerun:
        generated_summary_file = max(existing_summaries, key=os.path.getmtime)
        
    try:
        if generated_summary_file is None:
            tracker = Ocapi(
                subject_id=subject_id,
                session=session,
                config_file_path="config.yml",
                WEBCAM=None,
                VIDEO_INPUT=video_path,
                VIDEO_OUTPUT=None,
                TRACKING_DATA_LOG_FOLDER=output_log_dir
            )
            tracker.SHOW_ON_SCREEN_DATA = False
            
            tracker.sync_with_eeg_and_set_onsets(
                header_file=eeg_header_file,
                marker_file=eeg_marker_file,
                stimulus_description="Stimulus",
                events_of_interest=["S 21", "S 22", "S 23", "S 24"]
            )
            
            run_start_time = datetime.now()
            tracker.run()
            
            import time
            time.sleep(1)
            
            summary_pattern = os.path.join(output_log_dir, f"{subject_sess}_trial_summary*.csv")
            generated_files = glob.glob(summary_pattern)
            if not generated_files:
                raise FileNotFoundError(f"No trial summary files generated matching pattern {summary_pattern}")
                
            run_files = [f for f in generated_files if os.path.getmtime(f) > run_start_time.timestamp() - 5]
            if not run_files:
                raise FileNotFoundError(f"Could not find the trial summary file created during this run.")
            generated_summary_file = max(run_files, key=os.path.getmtime)
            
        # 4. Calculate Cohen's Kappa
        kappa_1, summary_1 = calculate_cohens_kappa(
            file_path1=rater1_file,
            file_path2=generated_summary_file,
            column_index1=0,
            column_index2=6,
            labels=[1, 2]
        )
        
        kappa_2, agreement_2 = None, None
        if os.path.exists(rater2_file):
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
            
            return {
                "status": "success",
                "subject": subject_sess,
                "trials": summary_1.get("total_observations", 0),
                "rater1_kappa": kappa_1,
                "rater1_agreement": agreement_1,
                "rater2_kappa": kappa_2 if kappa_2 is not None else "N/A",
                "rater2_agreement": agreement_2 if agreement_2 is not None else "N/A",
                "mean_kappa": mean_kappa
            }
        else:
            return {"status": "error", "message": f"Could not calculate Kappa for {subject_sess}"}
            
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"Error processing {subject_sess}: {e}\n{traceback.format_exc()}"}
    finally:
        gc.collect()

def run_validation():
    print("=" * 80)
    print("           OCAPI PARALLEL BATCH VALIDATION PIPELINE")
    print("=" * 80)
    
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
    
    target_subjects = None
    force_rerun = False
    
    if target_subjects:
        video_files = [f for f in video_files if any(sub in os.path.basename(f) for sub in target_subjects)]
    video_files = sorted(video_files)
    
    # Filter files that have rater 1 coding before scheduling tasks
    filtered_video_files = []
    for f in video_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        try:
            parts = base_name.split('_')
            subject_sess = f"{parts[0]}_{parts[1]}"
            rater1_file = os.path.join(rater_1_dir, f"{subject_sess}_VideoCoding.xlsx")
            if not os.path.exists(rater1_file):
                glob_matches = glob.glob(os.path.join(rater_1_dir, f"{subject_sess}*"))
                if not glob_matches:
                    continue
            filtered_video_files.append(f)
        except Exception:
            continue
            
    print(f"Total video files with human coding to process: {len(filtered_video_files)}")
    
    # Prepare task arguments
    tasks_args = []
    for idx, video_path in enumerate(filtered_video_files):
        tasks_args.append((
            idx,
            len(filtered_video_files),
            video_path,
            eeg_base_dir,
            rater_1_dir,
            rater_2_dir,
            output_log_dir,
            force_rerun
        ))
        
    results = []
    max_workers = 2 # Process 2 videos concurrently to speed up while keeping RAM low and safe
    
    print(f"Starting execution with {max_workers} processes...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video_task, args): args for args in tasks_args}
        
        completed = 0
        for future in as_completed(futures):
            args = futures[future]
            video_path = args[2]
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            completed += 1
            
            try:
                res = future.result()
                if res["status"] == "success":
                    print(f"[{completed}/{len(filtered_video_files)}] SUCCESS: {res['subject']} | Mean Kappa: {res['mean_kappa']:.4f}")
                    results.append(res)
                elif res["status"] == "skipped":
                    print(f"[{completed}/{len(filtered_video_files)}] SKIPPED: {base_name} ({res['message']})")
                else:
                    print(f"[{completed}/{len(filtered_video_files)}] ERROR: {base_name} ({res['message']})")
            except Exception as e:
                print(f"[{completed}/{len(filtered_video_files)}] EXCEPTION: {base_name} ({e})")
                
    print("\n" + "=" * 80)
    print("                       VALIDATION SUMMARY TABLE")
    print("=" * 80)
    if results:
        df_results = pd.DataFrame(results)
        # Drop status column from summary print
        print(df_results.drop(columns=["status"]).to_string(index=False))
        
        summary_csv_path = os.path.join(project_root, "validation_run_summary.csv")
        df_results.to_csv(summary_csv_path, index=False)
        print(f"\nDetailed summary saved to: {summary_csv_path}")
        
        overall_mean_kappa = df_results["mean_kappa"].mean()
        print(f"\nOverall Average Cohen's Kappa across all processed subjects: {overall_mean_kappa:.4f}")
    else:
        print("No subjects were successfully validated.")
    print("=" * 80)

if __name__ == "__main__":
    run_validation()
