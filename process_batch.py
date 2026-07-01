#!/usr/bin/env python
import argparse
import os
import pathlib
import sys


def process_batch(input_folder, output_folder, config_file, demo=False):
	"""
	This function contains the core video processing logic.
	It's called only when the script is running inside the target conda environment.
	"""
	# Import ocapi here to ensure it's imported by the
	# Python interpreter from the 'hgt' environment.
	from ocapi import get_data_path
	from ocapi.ocapi import Ocapi

	print(f"--- Running video processing in Python: {sys.executable} ---")

	if not os.path.isdir(input_folder):
		print(f"Error: Input folder '{input_folder}' not found.")
		return

	if not os.path.isdir(output_folder):
		print(f"Output folder '{output_folder}' not found. Creating it.")
		os.makedirs(output_folder, exist_ok=True)

	if not os.path.isfile(config_file):
		print(f"Error: Config file '{config_file}' not found.")
		return

	video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.mpeg', '.mpg']
	eeg_stimulus_description = "Stimulus"  # As seen in retrieve_head_gaze_data.py
	eeg_events_of_interest = ["S 21", "S 22", "S 23", "S 24"] # As seen in retrieve_head_gaze_data.py
	videos_processed_count = 0
	# --- New: Lists to track successes and failures ---
	successful_files = []
	failed_files = []

	input_path = pathlib.Path(input_folder)
	print(f"Scanning for videos in: {input_folder}")

	# Use glob to find all video files, similar to retrieve_head_gaze_data.py
	# We first check for videos in subfolders under 'trimmed', otherwise check direct folder contents
	video_files = [p for ext in video_extensions for p in input_path.glob(f"**/trimmed/*{ext}")]
	if not video_files:
		video_files = [p for ext in video_extensions for p in input_path.glob(f"*{ext}")]

	for video_input_path in sorted(video_files):
		if demo and videos_processed_count >= 1:
			print("\n--- Demonstration Mode: Stopped after processing 1 video ---")
			break
		base_name = video_input_path.stem
		print(f"\nProcessing video: {video_input_path}")
		
		# --- Enhanced Logic to handle Subject/Session and find EEG files ---
		try:
			subject_id, session, *_ = base_name.split('_')
		except ValueError:
			reason = f"Could not parse subject/session from filename '{video_input_path.name}'"
			print(f"  Warning: {reason}. Skipping.")
			failed_files.append((video_input_path.name, reason))
			continue

		# --- Robust EEG file finding, mirroring retrieve_head_gaze_data.py ---
		eeg_base_path = pathlib.Path(f"{get_data_path()}input/{subject_id}/{session}")
		try:
			# Find EEG Header File (.vhdr)
			header_pattern = f"{subject_id}_{session}*.vhdr"
			found_headers = list(eeg_base_path.glob(header_pattern))
			if not found_headers:
				raise FileNotFoundError(f"No EEG header file found in '{eeg_base_path}' matching '{header_pattern}'")
			if len(found_headers) > 1:
				raise FileNotFoundError(f"Multiple EEG header files found, please specify one: {found_headers}")
			eeg_header_file = found_headers[0]

			# Find EEG Marker File (.vmrk)
			marker_pattern = f"{subject_id}_{session}*.vmrk"
			found_markers = list(eeg_base_path.glob(marker_pattern))
			if not found_markers:
				raise FileNotFoundError(f"No EEG marker file found in '{eeg_base_path}' matching '{marker_pattern}'")
			if len(found_markers) > 1:
				raise FileNotFoundError(f"Multiple EEG marker files found, please specify one: {found_markers}")
			eeg_marker_file = found_markers[0]

		except FileNotFoundError as e:
			reason = str(e)
			print(f"  Warning: {reason}. Skipping video.")
			failed_files.append((video_input_path.name, reason))
			continue

		processed_video_filename = f"{base_name}_processed{video_input_path.suffix}"
		video_output_path = os.path.join(output_folder, processed_video_filename)
		tracking_data_log_folder = os.path.join(output_folder, "logs")
		os.makedirs(tracking_data_log_folder, exist_ok=True)

		print(f"  Subject ID: {subject_id}")
		print(f"  Session: {session}")
		print(f"  Output video will be: {video_output_path}")
		print(f"  Log folder will be: {tracking_data_log_folder}")

		try:
			tracker = Ocapi(
				subject_id=subject_id,
				session=session,
				config_file_path=config_file,
				WEBCAM=None, # Assuming batch processing is always from files
				VIDEO_INPUT=video_input_path,
				VIDEO_OUTPUT=video_output_path,
				TRACKING_DATA_LOG_FOLDER=tracking_data_log_folder,
			)

			# Perform EEG synchronization for each video
			print("  Synchronizing with EEG data...")
			tracker.sync_with_eeg_and_set_onsets(
				header_file=eeg_header_file,
				marker_file=eeg_marker_file,
				stimulus_description=eeg_stimulus_description,
				events_of_interest=eeg_events_of_interest
			)

			print("  Starting analysis...")
			tracker.run()
			videos_processed_count += 1
			successful_files.append(video_input_path.name)
			print(f"Finished processing: {video_input_path.name}")
		except IOError as e:
			reason = f"IOError: {e}"
			print(f"  Error processing {video_input_path.name}: {reason}. Skipping.")
			failed_files.append((video_input_path.name, reason))
		except Exception as e:
			reason = f"Unexpected error: {e}"
			print(f"  {reason} while processing {video_input_path.name}. Skipping video.")
			failed_files.append((video_input_path.name, reason))
			import traceback
			traceback.print_exc()

	# --- New: Write a summary log file ---
	summary_log_path = os.path.join(output_folder, "batch_summary.log")
	print(f"\n--- Batch Processing Summary ---")
	print(f"Successfully processed: {len(successful_files)} video(s)")
	print(f"Failed to process:    {len(failed_files)} video(s)")
	print(f"Summary log written to: {summary_log_path}")

	with open(summary_log_path, 'w') as f:
		f.write("--- Batch Processing Summary ---\n\n")
		f.write(f"Input Folder: {input_folder}\n")
		f.write(f"Output Folder: {output_folder}\n")
		f.write("-" * 30 + "\n\n")

		f.write(f"--- FAILED FILES ({len(failed_files)}) ---\n")
		if not failed_files:
			f.write("None\n")
		for filename, reason in failed_files:
			f.write(f"- {filename}: {reason}\n")
		f.write("\n")

		f.write(f"--- SUCCESSFULLY PROCESSED FILES ({len(successful_files)}) ---\n")
		if not successful_files:
			f.write("None\n")
		for filename in successful_files:
			f.write(f"- {filename}\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Batch process videos with ocapi and EEG synchronization.")
	parser.add_argument('-i', '--input', required=True, help="Folder containing the input videos (e.g., '.../trimmed/').")
	parser.add_argument('-o', '--output', required=True, help="Folder to save processed videos and logs.")
	parser.add_argument('-c', '--config', default="config.yml", help="Path to the configuration file (e.g., 'config.yml').")
	parser.add_argument('-d', '--demo', action='store_true', help="Run in demonstration mode (stop after processing 1 video).")

	args = parser.parse_args()

	process_batch(args.input, args.output, args.config, demo=args.demo)
