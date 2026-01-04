from typing import List
import ezc3d
import numpy as np
import json
from dtw import dtw
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from LMAHelpers import LMAHelper


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

    def calculate_score(self):
        markers_for_body_component = ["RASI", "RPSI", "LASI", "LPSI", "RANK", "LANK", "RWRA", "RWRB", "RSHO", "LWRA",
                                      "LWRB", "LSHO"]
        markers_for_effort_component = ["RFHD", "CLAV", "LFHD", "LBHD", "C7", "RBHD", "RASI", "RPSI", "LASI", "LPSI"]
        unique_markers = list(set(markers_for_body_component + markers_for_effort_component))

        # Perform global DTW
        aligned_original, aligned_user, alignment_path = self.align_sequences_globally(unique_markers)
        num_frames = len(alignment_path.index1)

        # Calculate scores for each component
        body_score = self.calculate_body_component(aligned_original, aligned_user, markers_for_body_component,
                                                   num_frames)
        effort_score = self.calculate_effort_component(aligned_original, aligned_user, markers_for_effort_component,
                                                       num_frames)

        print("Body score: ", body_score)
        print("Effort score: ", effort_score)

    @staticmethod
    def calculate_body_component(
            aligned_original: dict[str, np.ndarray], aligned_user: dict[str, np.ndarray],
            markers: List[str], num_frames: int
    ):
        """
        Calculate Body component for LMA using these features:
        - feet to hip distance
        - hands to shoulders distance
        - hands to hips distance
        - distance between feet
        Args:
            :param aligned_user:  {marker_name: array of shape (n_frames, 3)}
            :param aligned_original: {marker_name: array of shape (n_frames, 3)}
            :param markers: list of marker names
            :param num_frames: frame count
        """

        # Calculate the distances for each frame of the aligned sequences
        original_dist_body_vector = LMAHelper.calculate_distances(
            aligned_original, markers, num_frames, LMAHelper.calculate_distances_for_body_component)
        user_dist_body_vector = LMAHelper.calculate_distances(
            aligned_user, markers, num_frames, LMAHelper.calculate_distances_for_body_component)

        # Apply rolling window
        original_dist_body_features, user_dist_body_features = LMAHelper.calculate_rolling_window_features(
            original_dist_body_vector, user_dist_body_vector)

        # Calculate Pearson's correlation
        original_df_dist = pd.DataFrame(original_dist_body_features)
        user_df_dist = pd.DataFrame(user_dist_body_features)
        correlations_dist = original_df_dist.corrwith(user_df_dist, axis=0)

        # Apply Gaussian filter
        filtered_correlations_dist = gaussian_filter1d(correlations_dist, sigma=1)
        score = np.mean(filtered_correlations_dist)

        return score

    @staticmethod
    def calculate_effort_component(
            aligned_original: dict[str, np.ndarray], aligned_user: dict[str, np.ndarray],
            markers: List[str], num_frames: int
    ):
        """
        Calculate Effort component using these feature
        - head orientation
        - root joint velocity
        - root joint acceleration
        - root joint jerk
         Args:
            :param aligned_user:  {marker_name: array of shape (n_frames, 3)}
            :param aligned_original: {marker_name: array of shape (n_frames, 3)}
            :param markers: list of marker names
            :param num_frames: frame count
        """

        # Calculate head orientation distances for each frame
        original_dist_effort_vector = LMAHelper.calculate_distances(
            aligned_original, markers, num_frames, LMAHelper.calculate_distances_for_effort_component)
        user_dist_effort_vector = LMAHelper.calculate_distances(
            aligned_user, markers, num_frames, LMAHelper.calculate_distances_for_effort_component)

        # Calculate kinematic features for root joint
        original_root_joint_features = LMAHelper.calculate_root_joint_features(aligned_original)
        user_root_joint_features = LMAHelper.calculate_root_joint_features(aligned_user)

        # Apply rolling window
        original_dist_effort_features, user_dist_effort_features = LMAHelper.calculate_rolling_window_features(
            original_dist_effort_vector, user_dist_effort_vector)
        original_root_joint, user_root_joint = LMAHelper.calculate_rolling_window_features(
            original_root_joint_features, user_root_joint_features)

        # Calculate Pearson's correlation
        original_df_dist = pd.DataFrame(original_dist_effort_features)
        user_df_dist = pd.DataFrame(user_dist_effort_features)
        original_df_root_joint = pd.DataFrame(original_root_joint)
        user_df_root_joint = pd.DataFrame(user_root_joint)
        correlations_dist = original_df_dist.corrwith(user_df_dist, axis=0)
        correlations_root_joint = original_df_root_joint.corrwith(user_df_root_joint, axis=0)

        # Apply Gaussian filter
        filtered_correlations_dist = gaussian_filter1d(correlations_dist, sigma=1)
        filtered_correlations_root_joint = gaussian_filter1d(correlations_root_joint, sigma=1)
        combined_correlations = pd.concat(
            [pd.Series(filtered_correlations_dist), pd.Series(filtered_correlations_root_joint)]
        )
        score = combined_correlations.mean()

        return score


grade = LMAGrader("../sequence_with_kalman_online.json", "../m_krok_podstaw_uklon_polonez_1b.3.c3d")
grade.calculate_score()
