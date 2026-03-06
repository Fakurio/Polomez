import multiprocessing as mp
import queue
import time
import numpy as np
import pandas as pd
import signal

from vicon_dssdk import ViconDataStream
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
from UDPStreamer import UDPStreamer


def acquisition_process(vicon_host, frames_queue, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("[Acquisition] Process started.")

    client = ViconDataStream.Client()

    try:
        print(f"[Acquisition] Connecting to Vicon at {vicon_host}...")
        client.Connect(vicon_host)
        while not client.IsConnected() and not stop_event.is_set():
            time.sleep(1)

        if stop_event.is_set():
            return

        print("[Acquisition] Connected.")

        try:
            client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
        except Exception:
            print("[Acquisition] Could not set StreamMode to ServerPush.")

        client.EnableMarkerData()
        client.EnableSegmentData()

        # Buffer Flush
        print("[Acquisition] Flushing data buffer...")
        flush_end_time = time.time() + 0.5
        dropped_frames = 0
        while time.time() < flush_end_time:
            client.GetFrame()
            dropped_frames += 1
        print(f"[Acquisition] Buffer flushed ({dropped_frames} frames discarded).")

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
                    translation_data = client.GetMarkerGlobalTranslation(subject, marker_name)

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

    except Exception as e:
        print(f"[Acquisition] Error: {e}")
    finally:
        frames_queue.put(None)

        if client.IsConnected():
            client.Disconnect()
        print("[Acquisition] Process finished.")


def processing_process(mode, model_path, marker_groups, frames_queue, processed_queue,
                       processed_frames_to_send_queue, stop_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("[Processing] Process started.")

    if mode == "kalman":
        print("[Processing] Initializing Kalman Estimator...")
        estimator = KalmanEstimator(marker_groups)
    else:
        print("[Processing] Initializing transparent mode...")

    frame_number = 0
    while True:
        frame_data = frames_queue.get()

        if frame_data is None:
            break

        frame_number += 1
        if mode == "kalman":
            estimated_frame = estimator.estimate_frame(frame_data)
        else:
            estimated_frame = frame_data

        processed_queue.put(estimated_frame)
        processed_frames_to_send_queue.put(estimated_frame)

        if frame_number % 100 == 0:
            print(f"[Processing] Frame: {frame_number}")

    processed_frames_to_send_queue.put(None)
    processed_queue.put(None)

    print("[Processing] Process finished.")


def streamer_process(pc2_ip, port_llm, processed_frames_to_send_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("[Streaming] Process started.")

    streamer = UDPStreamer(pc2_ip, port_llm)

    frame_number = 0
    while True:
        frame_data = processed_frames_to_send_queue.get()

        if frame_data is None:
            break

        frame_number += 1
        streamer.send(frame_data)

        if frame_number % 100 == 0:
            print("Streaming frame: ", frame_data)
            print(f"[Streaming] Frame: {frame_number}")

    print("[Streaming] Process finished.")


def save_to_csv(processed_data, filename="output_sequence.csv"):
    print(f"Saving {len(processed_data)} frames to {filename}...")

    if not processed_data:
        print("No data to save.")
        return

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
    MODE = "none"
    MODEL_PATH = ""
    VICON_HOST = "localhost"
    PC2_IP = "127.0.0.1"
    PORT_LLM = 5000

    raw_frames_queue = mp.Queue()
    processed_frames_queue = mp.Queue()
    processed_frames_to_send_queue = mp.Queue()

    stop_event = mp.Event()

    acq_proc = mp.Process(target=acquisition_process,
                          args=(VICON_HOST, raw_frames_queue, stop_event))
    proc_proc = mp.Process(target=processing_process,
                           args=(MODE, MODEL_PATH, MARKER_GROUPS, raw_frames_queue,
                                 processed_frames_queue, processed_frames_to_send_queue,
                                 stop_event))
    udp_proc = mp.Process(target=streamer_process,
                          args=(PC2_IP, PORT_LLM, processed_frames_to_send_queue))

    final_frames_list = []
    try:
        acq_proc.start()
        proc_proc.start()
        udp_proc.start()

        print("System running. Press Ctrl+C to stop.")

        while True:
            try:
                frame = processed_frames_queue.get(timeout=0.1)

                if frame is None:
                    break

                final_frames_list.append(frame)
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print("\nStopping capture... Please wait. Processing remaining frames in queue...")
        stop_event.set()

        while True:
            frame = processed_frames_queue.get()
            if frame is None:
                break
            final_frames_list.append(frame)

    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()

    finally:
        if acq_proc.is_alive():
            acq_proc.join()
        if proc_proc.is_alive():
            proc_proc.join()
        if udp_proc.is_alive():
            udp_proc.join()

        save_to_csv(final_frames_list, "captured_data/captured_sequence_estimated.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
