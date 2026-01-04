from vicon_dssdk import ViconDataStream
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
import numpy as np
import threading
import queue
import time
import pandas as pd
import sys


def acquisition_thread(client, frames_queue, stop_event):
    print("Vicon acquisition thread started. Press Ctrl+C to stop.")
    frame_counter = 0

    while not stop_event.is_set():
        if not client.GetFrame():
            continue

        frame_counter += 1
        subjects = client.GetSubjectNames()
        current_frame = {}

        for subject in subjects:
            markers = client.GetMarkerNames(subject)
            for marker in markers:
                marker_name = marker[0]

                # GetGlobalTranslation returns (X, Y, Z, Occluded)
                translation_data = client.GetMarkerGlobalTranslation(subject, marker_name)

                # Check if data is valid
                if translation_data:
                    x, y, z = translation_data[0], translation_data[1], translation_data[2]
                    is_occluded = translation_data[3]

                    if is_occluded:
                        # KalmanEstimator expects NaNs for missing/occluded data
                        current_frame[marker_name] = np.full(3, np.nan)
                    else:
                        current_frame[marker_name] = np.array([x, y, z])
                else:
                    current_frame[marker_name] = np.full(3, np.nan)

        frames_queue.put(current_frame)

        if frame_counter % 300 == 0:
            print(f"Acquired #Frame: {frame_counter}")

    print("Vicon acquisition thread finished.")


def processing_thread(frames_queue, processed_queue, estimator, stop_event):
    print("Processing thread started.")
    frame_number = 0

    while not stop_event.is_set() or not frames_queue.empty():
        try:
            frame_data = frames_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_number += 1
        estimated_frame = estimator.estimate_frame(frame_data)
        processed_queue.put(estimated_frame)

        if frame_number % 300 == 0:
            print(f"Estimated #Frame: {frame_number}")

    print("Processing thread finished.")


def save_to_csv(processed_data, filename="output_sequence.csv"):
    print(f"Saving {len(processed_data)} frames to {filename}...")

    if not processed_data:
        print("No data to save.")
        return

    # Flatten data for CSV format: LFHD_X, LFHD_Y, LFHD_Z,
    flattened_rows = []
    first_frame = processed_data[0]
    marker_names = sorted(list(first_frame.keys()))

    headers = []
    for m in marker_names:
        headers.extend([f"{m}_X", f"{m}_Y", f"{m}_Z"])

    for frame in processed_data:
        row = {}
        for marker in marker_names:
            pos = frame.get(marker)
            if pos is not None and not np.any(np.isnan(pos)):
                row[f"{marker}_X"] = pos[0]
                row[f"{marker}_Y"] = pos[1]
                row[f"{marker}_Z"] = pos[2]
            else:
                # Handle cases where estimation might have failed (leave empty or 0)
                # Typically pandas handles missing keys as NaN
                pass
        flattened_rows.append(row)

    df = pd.DataFrame(flattened_rows, columns=headers)
    df.to_csv(filename, index=False)
    print("Done.")


if __name__ == "__main__":
    raw_frames_queue = queue.Queue()
    processed_frames_queue = queue.Queue()
    stop_event = threading.Event()
    estimator = KalmanEstimator(MARKER_GROUPS)

    client = ViconDataStream.Client()
    try:
        client.Connect("localhost")
        while not client.IsConnected():
            print("Connecting to Vicon...")
            time.sleep(1)
        print("Connected to datastream")

        client.EnableMarkerData()
        client.EnableSegmentData()

        acq_thread = threading.Thread(target=acquisition_thread,
                                      args=(client, raw_frames_queue, stop_event))
        proc_thread = threading.Thread(target=processing_thread,
                                       args=(raw_frames_queue, processed_frames_queue, estimator, stop_event))

        acq_thread.start()
        proc_thread.start()

        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping capture...")
        stop_event.set()

    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()

    finally:
        if 'acq_thread' in locals() and acq_thread.is_alive():
            acq_thread.join()
        if 'proc_thread' in locals() and proc_thread.is_alive():
            proc_thread.join()
        if client.IsConnected():
            client.Disconnect()

        # Collect results
        final_frames_list = []
        while not processed_frames_queue.empty():
            final_frames_list.append(processed_frames_queue.get())

        # Save to CSV
        save_to_csv(final_frames_list, "captured_sequence_estimated.csv")
