import ezc3d
import numpy as np
import json
from dtw import dtw
import pandas as pd


class LMAGrader:
    def __init__(self, user_sequence_file: str, original_sequence_file: str):
        self.original_sequence = ezc3d.c3d(original_sequence_file)
        self.user_sequence = json.load(open(user_sequence_file, "r"))
        self.original_sequence_len = self.original_sequence["header"]["points"]["last_frame"] + 1
        self.user_sequence_len = len(self.user_sequence)

    def get_markers_data_from_original_sequence(self, marker_names: list[str]):
        all_marker_names = self.original_sequence['parameters']['POINT']['LABELS']['value']
        marker_indices = [all_marker_names.index(marker) for marker in marker_names]

        marker_data = {marker: [] for marker in marker_names}
        for i in range(self.original_sequence_len):
            frame_data = self.original_sequence["data"]["points"][:3, marker_indices, i]
            for j, marker in enumerate(marker_names):
                x, y, z = frame_data[:, j]
                marker_data[marker].append(np.array([x, y, z]))

        for marker in marker_names:
            marker_data[marker] = np.array(marker_data[marker])

        return marker_data

    def get_markers_data_from_user_sequence(self, marker_names: list[str]):
        marker_data = {marker: [] for marker in marker_names}

        for _, markers in self.user_sequence.items():
            for marker in marker_names:
                marker_data[marker].append(np.array(markers[marker]))

        for marker in marker_names:
            marker_data[marker] = np.array(marker_data[marker])

        return marker_data

    @staticmethod
    def calculate_distances_for_body_component(markers_data: dict[str, np.ndarray]):
        """
        Args:
            markers_data: {marker_name: [x, y, z]}
        """

        # Feet to hip distance
        r_hip_center = (markers_data["RASI"] + markers_data["RPSI"]) / 2.0
        l_hip_center = (markers_data["LASI"] + markers_data["LPSI"]) / 2.0
        feet_to_hip_dist = (np.linalg.norm(r_hip_center - markers_data["RANK"]) + np.linalg.norm(
            l_hip_center - markers_data["LANK"])) / 2.0

        # Hands to shoulders distance
        r_hand_center = (markers_data["RWRA"] + markers_data["RWRB"]) / 2.0
        l_hand_center = (markers_data["LWRA"] + markers_data["LWRB"]) / 2.0
        hands_to_shoulder_dist = (np.linalg.norm(r_hand_center - markers_data["RSHO"]) + np.linalg.norm(
            l_hand_center - markers_data["LSHO"])) / 2.0

        # Hands to hips distance
        hands_to_hip_dist = (np.linalg.norm(r_hand_center - r_hip_center) + np.linalg.norm(
            l_hand_center - l_hip_center)) / 2.0

        # Distance between feet
        dist_between_feet = np.linalg.norm(markers_data["LANK"] - markers_data["RANK"])

        return np.array([feet_to_hip_dist, hands_to_shoulder_dist, hands_to_hip_dist, dist_between_feet])

    def align_sequences_globally(self, marker_names: list[str]):
        """
        Performs a single, global DTW alignment on combined marker data.
        """
        original_markers_data = self.get_markers_data_from_original_sequence(marker_names)
        user_markers_data = self.get_markers_data_from_user_sequence(marker_names)

        sequence_len = min(self.original_sequence_len, self.user_sequence_len)

        # Concatenate the marker data horizontally for the global DTW
        original_combined = np.concatenate(
            [original_markers_data[marker][:sequence_len] for marker in marker_names], axis=1
        )
        user_combined = np.concatenate(
            [user_markers_data[marker][:sequence_len] for marker in marker_names], axis=1
        )

        alignment = dtw(original_combined, user_combined, keep_internals=True)

        # Get the aligned sequences for each individual marker
        aligned_original_sequences = {}
        aligned_user_sequences = {}
        for marker in marker_names:
            original_data = original_markers_data[marker]
            user_data = user_markers_data[marker]

            aligned_original_sequences[marker] = original_data[alignment.index1]
            aligned_user_sequences[marker] = user_data[alignment.index2]

        return aligned_original_sequences, aligned_user_sequences, alignment

    def calculate_body_component(self):
        """
        Calculate Body component for LMA using these features:
        - feet to hip distance
        - hands to shoulders distance
        - hands to hips distance
        - distance between feet
        """
        markers = ["RASI", "RPSI", "LASI", "LPSI", "RANK", "LANK", "RWRA", "RWRB", "RSHO", "LWRA", "LWRB", "LSHO"]

        # Perform global DTW
        aligned_original, aligned_user, alignment_path = self.align_sequences_globally(markers)
        num_frames = len(alignment_path.index1)

        # Calculate the distances for each frame of the aligned sequences
        original_dist_vector = []
        user_dist_vector = []
        for i in range(num_frames):
            original_frame_data = {marker: aligned_original[marker][i] for marker in markers}
            user_frame_data = {marker: aligned_user[marker][i] for marker in markers}

            original_dist_vector.append(self.calculate_distances_for_body_component(original_frame_data))
            user_dist_vector.append(self.calculate_distances_for_body_component(user_frame_data))
        # Shape (num_frames, 4)
        original_dist_vector = np.array(original_dist_vector)
        user_dist_vector = np.array(user_dist_vector)

        # Apply rolling window to calculate features
        window_size = 35
        window_step = 1
        # Shape (num_of_windows, num_of_feature * num_of_distances)
        original_features = []
        user_features = []
        for i in range(0, num_frames - window_size + 1, window_step):
            original_window = original_dist_vector[i:i + window_size]
            user_window = user_dist_vector[i:i + window_size]
            # Shape (num_of_feature * num_of_distances,) -> (16, )
            original_window_features = np.array([
                np.max(original_window, axis=0),
                np.min(original_window, axis=0),
                np.mean(original_window, axis=0),
                np.std(original_window, axis=0)
            ]).flatten()

            user_window_features = np.array([
                np.max(user_window, axis=0),
                np.min(user_window, axis=0),
                np.mean(user_window, axis=0),
                np.std(user_window, axis=0)
            ]).flatten()

            original_features.append(original_window_features)
            user_features.append(user_window_features)

        # Calculate Pearson's correlation
        original_df = pd.DataFrame(original_features)
        user_df = pd.DataFrame(user_features)
        correlations = original_df.corrwith(user_df, axis=0)
        correlation_score = correlations.mean()

        print(f"Body Component Correlation Score: {correlation_score}")
        return correlation_score


grade = LMAGrader("sequence_with_kalman_online.json", "m_krok_podstaw_uklon_polonez_1b.3.c3d")
grade.calculate_body_component()
