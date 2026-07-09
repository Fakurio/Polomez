import multiprocessing as mp
import queue
import time
import numpy as np
import signal
import socket
import json

from vicon_dssdk import ViconDataStream
from KalmanEstimatorv2 import KalmanEstimator
from marker_groups import MARKER_GROUPS
from UDPStreamer import UDPStreamer


def acquisition_process(vicon_host, frames_queue, stop_event, session_active_event):
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

        frame_counter = 0
        session_was_active = False

        while not stop_event.is_set():
            session_is_active = session_active_event.is_set()
            if not session_is_active and session_was_active:
                print("[Acquisition] Session ended. Flushing Vicon buffer...")
                flush_end_time = time.time() + 0.5
                dropped_frames = 0
                while time.time() < flush_end_time:
                    if client.GetFrame():
                        dropped_frames += 1
                print(f"[Acquisition] Buffer flushed ({dropped_frames} frames discarded). Idling...")
            elif session_is_active and not session_was_active:
                print("[Acquisition] Session started. Resuming frame acquisition...")
                frame_counter = 0

            session_was_active = session_is_active

            if not client.GetFrame():
                continue
            if not session_is_active:
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


def processing_process(mode, marker_groups, frames_queue,
                       processed_frames_to_send_queue, stop_event, session_active_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("[Processing] Process started.")

    if mode == "kalman":
        print("[Processing] Pre-loading Numba cache from disk...")
        _dummy = KalmanEstimator(marker_groups)
        del _dummy
        print("[Processing] Cache loaded.")

    estimator = None
    session_was_active = False
    frame_number = 0

    while not stop_event.is_set():
        session_is_active = session_active_event.is_set()

        if session_is_active and not session_was_active:
            print("[Processing] New session started.")
            if mode == "kalman":
                print("[Processing] Initializing fresh Kalman Estimator...")
                estimator = KalmanEstimator(marker_groups)
            else:
                print("[Processing] Transparent mode active.")
        elif not session_is_active and session_was_active:
            print("[Processing] Session ended. Destroying old estimator state...")
            estimator = None
            frame_number = 0

        session_was_active = session_is_active

        if not session_is_active:
            time.sleep(0.01)
            continue

        try:
            frame_data = frames_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if frame_data is None:
            break

        frame_number += 1
        if mode == "kalman" and estimator is not None:
            estimated_frame = estimator.estimate_frame(frame_data)
        else:
            estimated_frame = frame_data

        # processed_queue.put(estimated_frame)
        processed_frames_to_send_queue.put(estimated_frame)

        if frame_number % 100 == 0:
            print(f"[Processing] Frame: {frame_number}")

    processed_frames_to_send_queue.put(None)

    print("[Processing] Process finished.")


def streamer_process(pc2_ip, port_llm, processed_frames_to_send_queue, stop_event, session_active_event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("[Streaming] Process started.")

    streamer = UDPStreamer(pc2_ip, port_llm)
    session_was_active = False
    frame_number = 0

    while not stop_event.is_set():
        session_is_active = session_active_event.is_set()
        if not session_is_active and session_was_active:
            frame_number = 0
        session_was_active = session_is_active

        if not session_is_active:
            time.sleep(0.01)
            continue

        try:
            frame_data = processed_frames_to_send_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if frame_data is None:
            break

        frame_number += 1
        streamer.send(frame_data)

        if frame_number % 100 == 0:
            print(f"[Streaming] Frame sent: {frame_number}")

    print("[Streaming] Process finished.")


def flush_queue(q, queue_name="Queue"):
    flushed_count = 0
    while True:
        try:
            q.get(timeout=0.05)
            flushed_count += 1
        except queue.Empty:
            break

    print(f"[System] Drained {flushed_count} leftover frames from {queue_name}.")


def main():
    MODE = "kalman"
    VICON_HOST = "localhost"
    # PC2_IP = "172.30.56.4"
    PC2_IP = "127.0.0.1"
    PORT_LLM = 5005
    LISTEN_PORT = 5006

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', LISTEN_PORT))
    print(f"[System] Listening socket successfully bound to {LISTEN_PORT}.")

    raw_frames_queue = mp.Queue()
    processed_frames_to_send_queue = mp.Queue()
    stop_event = mp.Event()
    session_active_event = mp.Event()

    print("[System] Initializing background processes...")
    acq_proc = mp.Process(target=acquisition_process,
                          args=(VICON_HOST, raw_frames_queue, stop_event, session_active_event))
    proc_proc = mp.Process(target=processing_process,
                           args=(MODE, MARKER_GROUPS, raw_frames_queue, processed_frames_to_send_queue,
                                 stop_event, session_active_event))
    udp_proc = mp.Process(target=streamer_process,
                          args=(PC2_IP, PORT_LLM, processed_frames_to_send_queue, stop_event, session_active_event))
    acq_proc.start()
    proc_proc.start()
    udp_proc.start()
    print("[System] Processes running and idling. Waiting for 'session_start'...")

    try:
        sock.settimeout(0.1)
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                msg = json.loads(data.decode('utf-8'))

                if msg.get("type") == "session_start":
                    if not session_active_event.is_set():
                        print(f"[System] Received 'session_start' from {addr}. Resuming data flow...")
                        session_active_event.set()
                elif msg.get("type") == "session_end":
                    if session_active_event.is_set():
                        print(f"[System] Received 'session_end' from {addr}. Pausing data flow...")
                        session_active_event.clear()
                        time.sleep(0.1)
                        flush_queue(raw_frames_queue, "raw frames queue")
                        flush_queue(processed_frames_to_send_queue, "processed frames to send queue")
                        print(f"[System] Queues flushed.")

            except socket.timeout:
                pass
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"[System] Network Error: {e}")

    except KeyboardInterrupt:
        print("\n[System] Ctrl+C detected. Shutting down completely...")

    finally:
        stop_event.set()
        session_active_event.clear()

        flush_queue(raw_frames_queue, "raw frames queue")
        flush_queue(processed_frames_to_send_queue, "processed frames to send queue")

        print("[System] Joining processes...")
        for p in [acq_proc, proc_proc, udp_proc]:
            p.join(timeout=2)
            if p.is_alive():
                print(f"[System] Terminating stuck process: {p.name}")
                p.terminate()
                p.join()

        raw_frames_queue.close()
        processed_frames_to_send_queue.close()
        sock.close()
        print("[System] System successfully closed.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
