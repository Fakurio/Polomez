from vicon_dssdk import ViconDataStream
from KalmanEstimator import KalmanEstimator
from marker_groups import MARKER_GROUPS
import numpy as np
import threading
import queue
import json


def convert_to_json(obj):
    if isinstance(obj, (np.ndarray, np.ma.MaskedArray)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json(v) for k, v in obj.items()}
    else:
        return obj


def acquisition_thread(client, frames_queue, stop_event, frames_count):
    print("Vicon acquisition thread started.")
    frame_counter = 0
    while not stop_event.is_set() and frame_counter < frames_count:
        if not client.GetFrame():
            continue

        frame_counter += 1
        subjects = client.GetSubjectNames()
        current_frame = {}
        for subject in subjects:
            markers = client.GetMarkerNames(subject)
            for marker in markers:
                translation = client.GetMarkerGlobalTranslation(subject, marker[0])
                current_frame[marker[0]] = translation

        frames_queue.put((frame_counter, current_frame))
        if frame_counter % 300 == 0:
            print(f"Acquired #Frame: {frame_counter}")
    print("Vicon acquisition thread finished.")
    stop_event.set()


def processing_thread(frames_queue, processed_queue, estimator, stop_event):
    print("Processing thread started.")
    while not stop_event.is_set() or not frames_queue.empty():
        try:
            frame_number, frame_data = frames_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        estimated_frame = estimator.estimate_frame(frame_data)
        processed_queue.put((f"frame_{frame_number}", estimated_frame))
        if frame_number % 300 == 0:
            print(f"Estimated #frame: {frame_number}")
    print("Processing thread finished.")


if __name__ == "__main__":
    raw_frames_queue = queue.Queue()
    processed_frames_queue = queue.Queue()
    stop_event = threading.Event()
    estimator = KalmanEstimator(MARKER_GROUPS)

    # TODO: finalnie trza jakoś rozpoznować kiedy zaczać i skończyć łapać klatki
    # FRAMES_COUNT = 3905 #k_krok_podstaw_uklon_polonez_1.3
    FRAMES_COUNT = 3534  # m_krok_podstaw_uklon_polonez_1b.3

    client = ViconDataStream.Client()
    client.Connect("localhost")

    while not client.IsConnected():
        pass
    print("Connected to datastream")

    client.EnableMarkerData()

    acquisition_thread = threading.Thread(target=acquisition_thread,
                                          args=(client, raw_frames_queue, stop_event, FRAMES_COUNT))
    filter_thread = threading.Thread(target=processing_thread,
                                     args=(raw_frames_queue, processed_frames_queue, estimator, stop_event))
    acquisition_thread.start()
    filter_thread.start()

    acquisition_thread.join()
    filter_thread.join()

    print("\nDatastream collection and processing done")

    client.Disconnect()

    # Save processed frames to JSON
    final_frames = {}
    while not processed_frames_queue.empty():
        key, value = processed_frames_queue.get()
        final_frames[key] = value

    serializable_data = convert_to_json(final_frames)
    with open("sequence_with_kalman_online.json", "w") as file:
        json.dump(serializable_data, file)
