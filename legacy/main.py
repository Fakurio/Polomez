# threading

from vicon_dssdk import ViconDataStream
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
import numpy as np
import threading
import queue
import time
import pandas as pd
from estimators import KalmanEstimatorWrapper, LSTMEstimator
from UDPStreamer import UDPStreamer


def acquisition_thread(client, frames_queue, stop_event):
    print("[Thread] Acquisition started.")

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

                # GetGlobalTranslation returns ((X, Y, Z), Occluded)
                translation_data = client.GetMarkerGlobalTranslation(subject, marker_name)

                # Check if data is valid
                if translation_data:
                    coords, is_occluded = translation_data[0], translation_data[1]
                    x, y, z = coords[0], coords[1], coords[2]

                    if is_occluded:
                        current_frame[marker_name] = np.full(3, np.nan)
                    else:
                        current_frame[marker_name] = np.array([x, y, z])
                else:
                    current_frame[marker_name] = np.full(3, np.nan)

        frames_queue.put(current_frame)
        if frame_counter % 300 == 0:
            print(f"[Acquisition] Frame: {frame_counter}")

    print("[Thread] Acquisition finished.")


def processing_thread(frames_queue, processed_queue, processed_frames_to_send_queue, estimator,
                      stop_event, processing_done_event):
    print("[Thread] Processing started.")

    frame_number = 0
    while not stop_event.is_set() or not frames_queue.empty():
        try:
            frame_data = frames_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_number += 1
        estimated_frame = estimator.estimate_frame(frame_data)
        processed_queue.put(estimated_frame)
        processed_frames_to_send_queue.put(estimated_frame)

        if frame_number % 100 == 0:
            print(f"[Processing] Frame: {frame_number}")

    print("[Thread] Processing finished.")
    processing_done_event.set()


def streamer_thread(processed_frames_to_send_queue, streamer, processing_done_event):
    print("[Thread] Streaming started.")

    frame_number = 0
    while not processing_done_event.is_set() or not processed_frames_to_send_queue.empty():
        try:
            frame_data = processed_frames_to_send_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        frame_number += 1
        streamer.send(frame_data)
        if frame_number % 100 == 0:
            print(f"[Streaming] Frame: {frame_number}")

    print("[Thread] Streaming finished.")


def save_to_csv(processed_data, filename="output_sequence.csv"):
    print(f"Saving {len(processed_data)} frames to {filename}...")

    if not processed_data:
        print("No data to save.")
        return

    # Flatten data for CSV format: LFHD_X, LFHD_Y, LFHD_Z,
    flattened_rows = []
    marker_names = sorted(list(MARKER_GROUPS.keys()))

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
                row[f"{marker}_X"] = np.nan
                row[f"{marker}_Y"] = np.nan
                row[f"{marker}_Z"] = np.nan
        flattened_rows.append(row)

    df = pd.DataFrame(flattened_rows, columns=headers)
    df.to_csv(filename, index=False)
    print("Done.")


def main():
    MODE = "kalman"
    MODEL_PATH = "best_lstm_bone_model.pth"
    VICON_HOST = "localhost"
    PC2_IP = "127.0.0.1"
    PORT_LLM = 5000

    # --- Setup Estimator ---
    if MODE == "kalman":
        print("Initializing Kalman Estimator...")
        core_estimator = KalmanEstimator(MARKER_GROUPS)
        estimator = KalmanEstimatorWrapper(core_estimator)
    else:
        print(f"Initializing LSTM Estimator (Model: {MODEL_PATH})...")
        estimator = LSTMEstimator(model_path=MODEL_PATH, root_marker="LASI")

    # --- Setup Queues & Vicon ---
    raw_frames_queue = queue.Queue()
    processed_frames_queue = queue.Queue()
    processed_frames_to_send_queue = queue.Queue()
    stop_event = threading.Event()
    processing_done_event = threading.Event()

    client = ViconDataStream.Client()
    streamer = UDPStreamer(PC2_IP, PORT_LLM)

    try:
        print(f"Connecting to Vicon at {VICON_HOST}...")
        client.Connect(VICON_HOST)
        while not client.IsConnected():
            print("...")
            time.sleep(1)
        print("Connected.")

        try:
            client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
            print("Stream mode set to ServerPush.")
        except Exception:
            print("Could not set StreamMode to ServerPush.")

        client.EnableMarkerData()
        client.EnableSegmentData()

        # --- 2. Buffer Flush ---
        print("Flushing data buffer...")
        flush_end_time = time.time() + 0.5
        dropped_frames = 0
        while time.time() < flush_end_time:
            client.GetFrame()
            dropped_frames += 1
        print(f"Buffer flushed ({dropped_frames} frames discarded). Starting threads.")

        # --- Start Threads ---
        acq_thread = threading.Thread(target=acquisition_thread,
                                      args=(client, raw_frames_queue, stop_event))
        proc_thread = threading.Thread(target=processing_thread,
                                       args=(raw_frames_queue,
                                             processed_frames_queue,
                                             processed_frames_to_send_queue,
                                             estimator, stop_event, processing_done_event))
        udp_thread = threading.Thread(target=streamer_thread,
                                      args=(processed_frames_to_send_queue, streamer, processing_done_event))

        acq_thread.start()
        proc_thread.start()
        udp_thread.start()

        # Keep main thread alive
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping capture...")
        stop_event.set()

    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()

    finally:
        # Cleanup
        if 'acq_thread' in locals() and acq_thread.is_alive():
            acq_thread.join()
        if 'proc_thread' in locals() and proc_thread.is_alive():
            proc_thread.join()
        if 'streamer_thread' in locals() and streamer_thread.is_alive():
            streamer_thread.join()
        if client.IsConnected():
            client.Disconnect()

        # Save Results
        final_frames_list = []
        while not processed_frames_queue.empty():
            final_frames_list.append(processed_frames_queue.get())
        save_to_csv(final_frames_list, "captured_data/captured_sequence_estimated.csv")


if __name__ == "__main__":
    main()
