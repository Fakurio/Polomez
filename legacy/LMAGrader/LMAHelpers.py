from typing import List, Callable
import numpy as np
from numpy import ndarray


class LMAHelper:
    @staticmethod
    def calculate_rolling_window_features(original_vector: ndarray, user_vector: ndarray, window_size=35,
                                          window_step=1):
        """
        Args:
            :param original_vector: array in shape (n_frames, n_features)
            :param user_vector: array in shape (n_frames, n_features)
            :param window_step: rolling window step
            :param window_size: rolling widow size
        """

        original_features = []
        user_features = []
        num_frames = original_vector.shape[0]

        for i in range(0, num_frames - window_size + 1, window_step):
            original_window = original_vector[i:i + window_size]
            user_window = user_vector[i:i + window_size]

            # Calculate statistical features for the original data window
            original_window_features = np.array([
                np.max(original_window, axis=0),
                np.min(original_window, axis=0),
                np.mean(original_window, axis=0),
                np.std(original_window, axis=0)
            ]).flatten()

            # Calculate statistical features for the user data window
            user_window_features = np.array([
                np.max(user_window, axis=0),
                np.min(user_window, axis=0),
                np.mean(user_window, axis=0),
                np.std(user_window, axis=0)
            ]).flatten()

            original_features.append(original_window_features)
            user_features.append(user_window_features)

        return original_features, user_features

    @staticmethod
    def calculate_distances(
            aligned_data: dict[str, np.ndarray], markers: List[str], num_frames: int, calculation_method: Callable
    ):
        """
        Calculates distances for whole sequence using calculation_method function
        Args:
            :param aligned_data: {marker_name: array of shape (n_frames, 3)}
            :param markers: list of marker names
            :param num_frames: frame count
            :param calculation_method: one of functions calculate_distance_for_...
        """
        dist_vector = []
        for i in range(num_frames):
            frame_data = {marker: aligned_data[marker][i] for marker in markers}
            dist_vector.append(calculation_method(frame_data))

        return np.array(dist_vector)

    @staticmethod
    def calculate_distances_for_body_component(markers_data: dict[str, np.ndarray]):
        """
        Calculates distances for one frame
        Args:
            :param markers_data: {marker_name: [x, y, z]}
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

    @staticmethod
    def calculate_distances_for_effort_component(markers_data: dict[str, np.ndarray]):
        """
        Calculates distances for one frame
        Args:
           :param markers_data: {marker_name: [x, y, z]}
        """

        # Calculate distances for head orientation
        dist_l_forehead_chest = np.linalg.norm(markers_data["LFHD"] - markers_data["CLAV"])
        dist_r_forehead_chest = np.linalg.norm(markers_data["RFHD"] - markers_data["CLAV"])
        dist_l_backhead_nape = np.linalg.norm(markers_data["LBHD"] - markers_data["C7"])
        dist_r_backhead_nape = np.linalg.norm(markers_data["RBHD"] - markers_data["C7"])

        return dist_l_forehead_chest, dist_r_forehead_chest, dist_l_backhead_nape, dist_r_backhead_nape

    @staticmethod
    def calculate_root_joint_features(markers_data: dict[str, np.ndarray]):
        """
        Calculates features for whole sequence
        Args:
            :param markers_data: {marker_name: array in shape (n_frames, 3)}
        """
        # Calculate the root joint position as mean of hips markers
        root_joint_positions = (
                                       markers_data["RASI"] +
                                       markers_data["RPSI"] +
                                       markers_data["LASI"] +
                                       markers_data["LPSI"]
                               ) / 4.0

        # Calculate kinematic magnitudes
        velocity = root_joint_positions[1:] - root_joint_positions[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        jerk = acceleration[1:] - acceleration[:-1]

        # Align sequences to a common length
        min_len = min(len(velocity), len(acceleration), len(jerk))
        velocity = velocity[:min_len]
        acceleration = acceleration[:min_len]
        jerk = jerk[:min_len]

        # Combine the magnitudes into a single feature vector for each frame
        kinematic_features = np.column_stack((velocity, acceleration, jerk))

        return kinematic_features
