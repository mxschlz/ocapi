import cv2 as cv
import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from ocapi.ocapi import Ocapi


class PlottingOcapi(Ocapi):
    """
    A subclass of Ocapi that extends the tracking loop to also plot
    head pose and eye gaze data in real-time using matplotlib, and saves
    both the webcam output and the plot animation to video files, combined
    horizontally, starting ONLY after manual calibration (pressing 'c').
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Determine history length (how many frames to display on the plot)
        self.history_length = 100
        
        # Initialize data buffers with zeros
        self.pitch_history = deque([0] * self.history_length, maxlen=self.history_length)
        self.yaw_history = deque([0] * self.history_length, maxlen=self.history_length)
        self.roll_history = deque([0] * self.history_length, maxlen=self.history_length)
        
        self.ldx_history = deque([0] * self.history_length, maxlen=self.history_length)
        self.ldy_history = deque([0] * self.history_length, maxlen=self.history_length)

        self.rdx_history = deque([0] * self.history_length, maxlen=self.history_length)
        self.rdy_history = deque([0] * self.history_length, maxlen=self.history_length)

        # Set up matplotlib for real-time updating
        plt.ion()
        # Make the figure dimensions suitable for combining horizontally with the webcam feed
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.canvas.manager.set_window_title('Live Head Pose & Gaze')
        
        # X-axis points
        self.x_data = list(range(self.history_length))
        
        # Set up Head Pose subplot
        self.line_pitch, = self.ax1.plot(self.x_data, self.pitch_history, 'b-', label='Pitch')
        self.line_yaw,   = self.ax1.plot(self.x_data, self.yaw_history, 'g-', label='Yaw')
        #self.line_roll,  = self.ax1.plot(self.x_data, self.roll_history, 'r-', label='Roll')
        self.ax1.set_ylim(-60, 60)
        self.ax1.set_ylabel('Degrees')
        self.ax1.set_title('Head Pose')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True)
        
        # Set up Gaze subplot (Relative Eye Positions)
        self.line_ldx, = self.ax2.plot(self.x_data, self.ldx_history, 'm-', label='Left Dx')
        self.line_ldy, = self.ax2.plot(self.x_data, self.ldy_history, 'c-', label='Left Dy')
        self.line_rdx, = self.ax2.plot(self.x_data, self.rdx_history, 'm--', label='Right Dx')
        self.line_rdy, = self.ax2.plot(self.x_data, self.rdy_history, 'c--', label='Right Dy')

        self.ax2.set_ylim(-30, 30)
        self.ax2.set_ylabel('Pixels')
        self.ax2.set_title('Relative Eye Positions (Dx, Dy)')
        self.ax2.legend(loc='upper right', ncol=2)
        self.ax2.grid(True)
        
        plt.tight_layout()
        self.plot_counter = 0

        # --- Combined Video Saving Setup ---
        self.combined_out = None
        self.has_started_recording = False
        
        # We don't initialize the writer here because we want to wait for calibration
        # and we need the dimensions of both the webcam frame and the plot frame
        if self.out:
            # We don't need the default single-video writer initialized by the parent class
            self.out.release()
            self.out = None

    def _init_combined_video_writer(self, webcam_frame, plot_frame):
        if not self.VIDEO_OUTPUT_BASE:
            return

        h1, w1 = webcam_frame.shape[:2]
        h2, w2 = plot_frame.shape[:2]
        
        # We will concatenate them horizontally
        target_h = max(h1, h2)
        target_w = w1 + w2

        # Create a new path for the combined video
        base, ext = os.path.splitext(self.VIDEO_OUTPUT_BASE)
        combined_output_path = f"{base}_combined{ext}"
        
        # Use measured FPS if available, to match original pacing,
        # since plotting significantly reduces the loop frame rate.
        if hasattr(self, 'fps_history') and len(self.fps_history) > 5:
            output_fps = sum(self.fps_history) / len(self.fps_history)
            logging.info(f"Using measured processing FPS: {output_fps:.2f}")
        else:
            output_fps = self.FPS
            logging.info(f"Using default FPS: {output_fps}")
            
        fourcc = cv.VideoWriter_fourcc(*getattr(self, 'OUTPUT_VIDEO_FOURCC', 'XVID').upper())
        
        self.combined_out = cv.VideoWriter(combined_output_path, fourcc, output_fps, (target_w, target_h))
        if self.combined_out.isOpened():
            logging.info(f"Started recording combined video to: {combined_output_path}")
        else:
            logging.error(f"Could not open combined video writer at {combined_output_path}")
            self.combined_out = None

    def _draw_on_screen_data(self, frame, landmarks, img_h, img_w, current_frame_time_ms):
        # Measure actual processing FPS
        if not hasattr(self, 'last_time'):
            self.last_time = time.time()
            self.fps_history = deque(maxlen=30)
        else:
            curr_time = time.time()
            dt = curr_time - self.last_time
            if dt > 0:
                self.fps_history.append(1.0 / dt)
            self.last_time = curr_time

        # Call the parent method to draw original OpenCV overlays
        super()._draw_on_screen_data(frame, landmarks, img_h, img_w, current_frame_time_ms)
        
        # Only update buffers and save frames IF we are calibrated
        if self.calibrated:
            # Clear buffers on the very first frame of calibration to start fresh
            if not self.has_started_recording:
                self.pitch_history = deque([0] * self.history_length, maxlen=self.history_length)
                self.yaw_history = deque([0] * self.history_length, maxlen=self.history_length)
                self.roll_history = deque([0] * self.history_length, maxlen=self.history_length)
                self.ldx_history = deque([0] * self.history_length, maxlen=self.history_length)
                self.ldy_history = deque([0] * self.history_length, maxlen=self.history_length)
                self.rdx_history = deque([0] * self.history_length, maxlen=self.history_length)
                self.rdy_history = deque([0] * self.history_length, maxlen=self.history_length)
                logging.info("Calibration confirmed. Starting data recording and plotting...")

            # Collect data for plots
            self.pitch_history.append(self.adj_pitch)
            self.yaw_history.append(self.adj_yaw)
            self.roll_history.append(self.adj_roll)
                
            self.ldx_history.append(self.l_dx)
            self.ldy_history.append(self.l_dy)
            self.rdx_history.append(self.r_dx)
            self.rdy_history.append(self.r_dy)
            
            # Update plot data
            self.line_pitch.set_ydata(self.pitch_history)
            self.line_yaw.set_ydata(self.yaw_history)
            self.line_roll.set_ydata(self.roll_history)
            
            self.line_ldx.set_ydata(self.ldx_history)
            self.line_ldy.set_ydata(self.ldy_history)
            self.line_rdx.set_ydata(self.rdx_history)
            self.line_rdy.set_ydata(self.rdy_history)
            
            self.fig.canvas.draw_idle()
            
            # Convert the matplotlib figure to a numpy array for OpenCV
            s, (width, height) = self.fig.canvas.print_to_buffer()
            plot_img_rgba = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            plot_img_bgr = cv.cvtColor(plot_img_rgba, cv.COLOR_RGBA2BGR)

            self.fig.canvas.flush_events()
            
            # --- Combine Videos horizontally ---
            h1, w1 = frame.shape[:2]
            h2, w2 = plot_img_bgr.shape[:2]
            target_h = max(h1, h2)
            
            # Pad the frames if they have different heights
            webcam_padded = cv.copyMakeBorder(frame, 0, target_h - h1, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
            plot_padded = cv.copyMakeBorder(plot_img_bgr, 0, target_h - h2, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Concatenate horizontally
            combined_frame = np.hstack((webcam_padded, plot_padded))
            
            # Initialize writer on the first calibrated frame
            if not self.has_started_recording:
                self._init_combined_video_writer(webcam_padded, plot_padded)
                self.has_started_recording = True

            # Write the combined frame
            if self.combined_out:
                self.combined_out.write(combined_frame)

    def _write_video_frame(self, frame):
        # Override the parent method so it doesn't try to write to the default writer
        # Writing is now handled in _draw_on_screen_data after combining
        pass

    def _cleanup(self, finalize_data=True):
        # Call parent cleanup
        super()._cleanup(finalize_data)
        
        # Release the combined video writer
        if self.combined_out:
            self.combined_out.release()
            logging.info("Combined video writer released.")
        
        # Keep plot window open until manually closed after webcam feed stops
        plt.ioff()
        logging.info("Tracker closed. You can now examine the final plot window.")
        plt.show()


def run_webcam_demo():
    """
    Initializes and runs the PlottingOcapi tracker with live webcam input.
    """
    # --- Configuration ---
    # Use a config file specific to this demo, located in the same directory.
    config_file = os.path.join(os.path.dirname(__file__), "config_webcam_demo.yml")
    subject_id = "webcam_demo_subject"
    session_id = "S1"
    output_folder = f"output/{subject_id}"

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # --- Check for Config File ---
    if not os.path.exists(config_file):
        logging.error(f"Configuration file not found at '{config_file}'.")
        logging.error("Please ensure 'config_webcam_demo.yml' is in the 'demos' directory.")
        return

    logging.info("--- Starting Webcam Demo with Live Plots ---")
    logging.info(f"Using config: {config_file}")
    logging.info("A Matplotlib window and an OpenCV window will open.")
    logging.info("IMPORTANT: Video recording will ONLY start after you press 'c' to calibrate.")
    
    try:
        # --- Initialize the PlottingOcapi tracker ---
        tracker = PlottingOcapi(
            subject_id=subject_id,
            session=session_id,
            config_file_path=config_file,
            WEBCAM=0,
            VIDEO_INPUT=None,
            VIDEO_OUTPUT=os.path.join(output_folder, "webcam_output.mp4"),
            TRACKING_DATA_LOG_FOLDER=os.path.join(output_folder, "logs")
        )

        # --- Run the Tracker ---
        logging.info("Starting the tracker. Press 'q' in the OpenCV display window to quit.")
        tracker.run()

    except ImportError as e:
        logging.error(f"A required library is not installed: {e}")
        logging.error("Please ensure 'matplotlib' is installed: pip install matplotlib")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("--- Webcam Demo Finished ---")
        cv.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()