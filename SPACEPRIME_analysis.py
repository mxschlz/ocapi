#!/usr/bin/env python3
"""
Process all SPACEPRIME subjects and blocks using the Ocapi tracker.
"""

import os
import sys
from pathlib import Path

# Add the ocapi module to the path
sys.path.insert(0, str(Path(__file__).parent))

from ocapi.ocapi import Ocapi

# Configuration
BASE_PATH = r"\\storage.ipsy1.uni-luebeck.de\IPSY1\Projects\ac\Experiments\running_studies\SPACEPRIME\sourcedata\raw"
OUTPUT_BASE_PATH = r"\\storage.ipsy1.uni-luebeck.de\IPSY1\Projects\ac\Experiments\running_studies\OCAPI\output\SPACEPRIME"
CONFIG_FILE = Path(__file__).parent / "config_spaceprime.yml"

# Subject IDs: 108, 110, 112, ..., 174
SUBJECT_IDS = list(range(108, 176, 2))  # This generates [108, 110, 112, ..., 174]
NUM_BLOCKS = 10  # Blocks 0-9


def process_subject(subject_id):
    """Process all blocks for a given subject."""
    print(f"\n{'=' * 80}")
    print(f"STARTING SUBJECT: sub-{subject_id}")
    print(f"{'=' * 80}\n")

    subject_dir = Path(BASE_PATH) / f"sub-{subject_id}" / "headgaze"

    # Check if subject directory exists
    if not subject_dir.exists():
        print(f"ERROR: Subject directory does not exist: {subject_dir}")
        return False

    # Create output directory for this subject
    output_dir = Path(OUTPUT_BASE_PATH) / f"sub-{subject_id}" / "headgaze"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_blocks_success = True

    for block_id in range(NUM_BLOCKS):
        video_file = subject_dir / f"sub-{subject_id}_block-{block_id}.asf"

        print(f"\n--- Block {block_id}/{NUM_BLOCKS - 1} ---")
        print(f"Video: {video_file}")

        # Check if video file exists
        if not video_file.exists():
            print(f"WARNING: Video file not found: {video_file}")
            all_blocks_success = False
            continue

        try:
            # Create output file path
            output_video = str(output_dir / f"sub-{subject_id}_block-{block_id}_processed.mp4")

            print(f"Output: {output_video}")
            print("Initializing tracker...")

            # Initialize the Ocapi tracker
            tracker = Ocapi(
                subject_id=f"sub-{subject_id}",
                config_file_path=str(CONFIG_FILE),
                VIDEO_INPUT=str(video_file),
                VIDEO_OUTPUT=output_video,
                TRACKING_DATA_LOG_FOLDER=str(output_dir),
                session=None,
                WEBCAM=None
            )

            print("Running analysis...")
            # Run the tracker
            tracker.run()

            print(f"✓ Block {block_id} completed successfully")

        except Exception as e:
            print(f"✗ ERROR processing block {block_id}: {e}")
            import traceback
            traceback.print_exc()
            all_blocks_success = False
            continue

    print(f"\n{'=' * 80}")
    if all_blocks_success:
        print(f"✓ SUBJECT sub-{subject_id}: ALL BLOCKS COMPLETED SUCCESSFULLY")
    else:
        print(f"⚠ SUBJECT sub-{subject_id}: COMPLETED WITH WARNINGS/ERRORS")
    print(f"{'=' * 80}\n")

    return all_blocks_success


def main():
    """Main entry point."""
    print(f"\n{'=' * 80}")
    print("SPACEPRIME Batch Processing")
    print(f"{'=' * 80}")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Base path: {BASE_PATH}")
    print(f"Output base path: {OUTPUT_BASE_PATH}")
    print(f"Number of subjects: {len(SUBJECT_IDS)}")
    print(f"Subject IDs: {SUBJECT_IDS}")
    print(f"Blocks per subject: {NUM_BLOCKS}")
    print(f"Total video files: {len(SUBJECT_IDS) * NUM_BLOCKS}")
    print(f"{'=' * 80}\n")

    # Check if config file exists
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        return False

    # Check if base path exists
    if not Path(BASE_PATH).exists():
        print(f"ERROR: Base path does not exist: {BASE_PATH}")
        return False

    # Process each subject
    successful_subjects = []
    failed_subjects = []

    for subject_id in SUBJECT_IDS:
        try:
            success = process_subject(subject_id)
            if success:
                successful_subjects.append(subject_id)
            else:
                failed_subjects.append(subject_id)
        except Exception as e:
            print(f"FATAL ERROR processing subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_subjects.append(subject_id)

    # Print final summary
    print(f"\n{'=' * 80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Successful subjects: {len(successful_subjects)}")
    if successful_subjects:
        print(f"  {successful_subjects}")
    print(f"\nFailed subjects: {len(failed_subjects)}")
    if failed_subjects:
        print(f"  {failed_subjects}")
    print(f"{'=' * 80}\n")

    return len(failed_subjects) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

