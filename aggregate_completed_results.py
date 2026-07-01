#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inter_rater_reliability import calculate_cohens_kappa

def main():
    print("=" * 80)
    print("           AGGREGATING COMPLETED VALIDATION RESULTS")
    print("=" * 80)
    
    validation_log_dir = os.path.join(project_root, "validation_logs")
    rater_1_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_1"
    rater_2_dir = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/OCAPI/output/human_coding/rater_2"
    
    # Find all trial summary files generated in validation_logs
    summary_files = glob.glob(os.path.join(validation_log_dir, "*_trial_summary_*.csv"))
    if not summary_files:
        print("No trial summary files found in validation_logs.")
        return
        
    print(f"Found {len(summary_files)} completed trial summaries.")
    results = []
    
    for summary_path in sorted(summary_files):
        filename = os.path.basename(summary_path)
        # Extract subject and session from filename e.g. SCS048_A_trial_summary_20260621_212801.csv
        parts = filename.split('_')
        subject_id = parts[0]
        session = parts[1]
        subject_sess = f"{subject_id}_{session}"
        
        # Paths to human codings
        rater1_file = os.path.join(rater_1_dir, f"{subject_sess}_VideoCoding.xlsx")
        rater2_file = os.path.join(rater_2_dir, f"{subject_sess}_VideoCoding.xlsx")
        
        if not os.path.exists(rater1_file):
            # Try matching with glob
            glob_matches = glob.glob(os.path.join(rater_1_dir, f"{subject_sess}*"))
            if glob_matches:
                rater1_file = glob_matches[0]
            else:
                print(f"  Rater 1 coding file missing for {subject_sess}. Skipping.")
                continue
                
        # Calculate Cohen's Kappa against Rater 1
        kappa_1, summary_1 = calculate_cohens_kappa(
            file_path1=rater1_file,
            file_path2=summary_path,
            column_index1=0,
            column_index2=6,
            labels=[1, 2]
        )
        
        kappa_2, agreement_2 = None, None
        if os.path.exists(rater2_file):
            kappa_2, summary_2 = calculate_cohens_kappa(
                file_path1=rater2_file,
                file_path2=summary_path,
                column_index1=0,
                column_index2=6,
                labels=[1, 2]
            )
            if summary_2:
                agreement_2 = summary_2.get("observed_agreement_proportion", 0)
                
        if kappa_1 is not None and summary_1 is not None:
            agreement_1 = summary_1.get("observed_agreement_proportion", 0)
            mean_kappa = (kappa_1 + kappa_2) / 2.0 if kappa_2 is not None else kappa_1
            
            # Find the original mean kappa from batch_optimization_summary.csv to see the difference!
            orig_kappa = "N/A"
            batch_summary_path = os.path.join(project_root, "batch_optimization_summary.csv")
            if os.path.exists(batch_summary_path):
                try:
                    df_batch = pd.read_csv(batch_summary_path)
                    row = df_batch[df_batch['subject'] == subject_sess]
                    if not row.empty:
                        orig_kappa = float(row.iloc[0]['algo_mean_kappa'])
                except Exception:
                    pass
            
            results.append({
                "Subject": subject_sess,
                "Trials": summary_1.get("total_observations", 0),
                "R1 Kappa": f"{kappa_1:.4f}",
                "R1 Agree": f"{agreement_1:.4f}",
                "R2 Kappa": f"{kappa_2:.4f}" if kappa_2 is not None else "N/A",
                "Mean Kappa (New)": mean_kappa,
                "Mean Kappa (Old)": orig_kappa,
                "Improvement": mean_kappa - orig_kappa if orig_kappa != "N/A" else "N/A"
            })
            
    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # Save results
        df_results.to_csv("validation_completed_summary.csv", index=False)
        print("\nSummary saved to: validation_completed_summary.csv")
        
        # Calculate overall mean
        valid_means = [r['Mean Kappa (New)'] for r in results]
        overall_mean = sum(valid_means) / len(valid_means)
        print(f"\nOverall Average Cohen's Kappa for completed subjects: {overall_mean:.4f}")
    else:
        print("No results calculated.")
    print("=" * 80)

if __name__ == "__main__":
    main()
