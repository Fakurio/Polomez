from pykalman import KalmanFilter
import numpy as np


class KalmanEstimator:
    def __init__(self, marker_groups: dict[str, list[str]], n_dims: int = 3, dt: float = 1 / 100.0,
                 process_noise: float = 0.01,
                 observation_noise: float = 0.01):
        """
        Args:
            marker_groups: Dictionary {marker_name: [marker_names_in_group]}
            dt: Time step (Vicon has default 100 Hz)
            process_noise: Process noise variance
            observation_noise: Observation noise variance
        """
        self.marker_groups = marker_groups
        self.dt = dt
        self.n_dims = n_dims
        self.process_noise = process_noise
        self.observation_noise = observation_noise

        # Dictionary with marker positions in t-1 frame
        self.last_known_positions = {}

        # Dictionary with marker positions in t-2 frame
        self.second_last_known_positions = {}

        # Dictionary to store Kalman filters for each marker
        self.filters = {marker_name: self._create_kalman_filter() for marker_name in marker_groups.keys()}

    def _create_kalman_filter(self):
        """
        Create a Kalman filter for a single marker using a constant velocity model
        """
        # State transition matrix [x, y, z, vx, vy, vz]
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Observation matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance matrix
        Q = np.eye(self.n_dims * 2) * self.process_noise

        # Observation noise covariance matrix
        R = np.eye(self.n_dims) * self.observation_noise

        kf = KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=np.zeros(self.n_dims * 2),
            initial_state_covariance=np.eye(self.n_dims * 2)
        )

        return kf

    @staticmethod
    def _find_closest_point_on_circle(circle_center: np.ndarray, circle_radius: float, circle_normal: np.ndarray,
                                      point: np.ndarray):
        """
        Finds the closest point on a 3D circle to a given point

        Args:
            circle_center: The center of the circle
            circle_radius: The radius of the circle
            circle_normal: The normal vector of the circle
            point: Point coordinates
        """
        # Project point onto radical plane
        v_cc_point = point - circle_center
        offset = v_cc_point @ circle_normal
        p_proj = point - offset * circle_normal

        v_cc_proj = p_proj - circle_center
        if np.isclose(np.linalg.norm(v_cc_proj), 0):
            # Projected point lies in a circle center
            closest_point = point
        else:
            # The closest point lies on a path from circle center to projected point
            v_cc_proj_norm = v_cc_proj / np.linalg.norm(v_cc_proj)
            closest_point = circle_center + circle_radius * v_cc_proj_norm

        return closest_point

    def _estimate_two_visible(self, missing_marker: str, visible_markers_data: dict[str, np.ndarray]):
        """
        Estimates the missing marker position when exactly two markers in its group are visible

        Args:
            missing_marker: The name of the missing marker
            visible_markers_data: Dictionary {marker_name: [x, y, z]}
        """
        assert len(visible_markers_data) == 2, "This function requires exactly two visible markers in the group."

        m1 = missing_marker
        m2, m3 = list(visible_markers_data.keys())
        x2_t, x3_t = visible_markers_data.get(m2), visible_markers_data.get(m3)

        # Retrieve marker positions from previous frame (x_t-1)
        x1_t_minus_1 = self.last_known_positions.get(m1)
        x2_t_minus_1 = self.last_known_positions.get(m2)
        x3_t_minus_1 = self.last_known_positions.get(m3)

        if x1_t_minus_1 is None or x2_t_minus_1 is None or x3_t_minus_1 is None:
            print(f"Previous frame data incomplete for {m1}, {m2}, {m3}. Falling back to filter prediction.")
            return None

        # Calculate Dt-1 vectors
        D12_t_minus_1 = x2_t_minus_1 - x1_t_minus_1
        D13_t_minus_1 = x3_t_minus_1 - x1_t_minus_1

        # Calculate average position in current frame
        x_mean_t = ((x2_t - D12_t_minus_1) + (x3_t - D13_t_minus_1)) / 2.0

        # Calculate the distance between 2 spheres with centers at x2_t and x3_t
        # and radius |D12_t_minus_1| and |D13_t_minus_1| respectively
        d = np.linalg.norm(x3_t - x2_t)

        D12_radius = np.linalg.norm(D12_t_minus_1)
        D13_radius = np.linalg.norm(D13_t_minus_1)

        if not np.abs(D12_radius - D13_radius) < d < D12_radius + D13_radius:
            # No circular intersection
            return x_mean_t

        # Calculate the distance between a sphere center at x2_t and the intersection circle center
        h = (d ** 2 - D13_radius ** 2 + D12_radius ** 2) / (2 * d)

        inter_circle_radius_squared = D12_radius ** 2 - h ** 2
        if inter_circle_radius_squared < 0:
            # No real intersection
            return x_mean_t

        inter_circle_radius = np.sqrt(inter_circle_radius_squared)
        v_norm = (x3_t - x2_t) / d
        inter_circle_center = x2_t + h * v_norm

        closest_point = self._find_closest_point_on_circle(inter_circle_center, inter_circle_radius, v_norm, x_mean_t)

        return closest_point

    def _estimate_one_visible(self, missing_marker: str, visible_markers_data: dict[str, np.ndarray]):
        """
           Estimates the missing marker position when exactly one marker in its group is visible

           Args:
            missing_marker: The name of the missing marker
            visible_markers_data: Dictionary {marker_name: [x, y, z]}
        """
        assert len(visible_markers_data) == 1, "This function requires exactly one visible marker in the group."

        m1 = missing_marker
        m2 = list(visible_markers_data.keys())[0]
        x2_t = visible_markers_data.get(m2)

        # Retrieve marker positions from previous frame (x_t-1)
        x1_t_minus_1 = self.last_known_positions.get(m1)
        x2_t_minus_1 = self.last_known_positions.get(m2)

        if x1_t_minus_1 is None or x2_t_minus_1 is None:
            print(f"Previous frame data incomplete for {m1}, {m2}. Falling back to filter prediction.")
            return None

        D12_t_minus_1 = x2_t_minus_1 - x1_t_minus_1
        x_mean_t = x2_t - D12_t_minus_1

        return x_mean_t

    @staticmethod
    def _calculate_rigid_transform(src_pts: np.ndarray, dst_pts: np.ndarray):
        """
        Calculates rigid body transformation using the Kabsch algorithm to transform
        source points to destination points.

        Args:
            src_pts: Nx3 array of source points
            dst_pts: Nx3 array of destination points
        """
        src_centroid = np.mean(src_pts, axis=0)
        dst_centroid = np.mean(dst_pts, axis=0)

        src_centered = src_pts - src_centroid
        dst_centered = dst_pts - dst_centroid

        H = src_centered.T @ dst_centered

        U, S, Vt = np.linalg.svd(H)

        if np.linalg.det(Vt.T @ U.T) < 0.0:
            Vt[-1, :] *= -1.0

        R = Vt.T @ U.T
        t = dst_centroid - R @ src_centroid

        return R, t

    def _estimate_from_rigid_body(self, missing_marker: str, visible_markers_data: dict[str, np.ndarray]):
        """
        Estimates the missing marker position based on the rigid body transformation.

        Args:
            missing_marker: The name of the missing marker
            visible_markers_data: Dictionary {marker_name: [x, y, z]}
        """
        src_pts = []  # Reference points
        dst_pts = []  # Visible points from current frame

        for marker_name, current_pos in visible_markers_data.items():
            if self.last_known_positions.get(marker_name) is not None:
                src_pts.append(self.last_known_positions.get(marker_name))
                dst_pts.append(current_pos)

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        if len(src_pts) < 3:
            print("Not enough source points for rigid body transformation. Falling back to filter prediction.")
            return None

        R, t = self._calculate_rigid_transform(src_pts, dst_pts)

        # Transform the last known position of the missing marker
        if self.last_known_positions.get(missing_marker) is not None:
            missing_marker_ref_pos = self.last_known_positions.get(missing_marker)
            estimated_pos = R @ missing_marker_ref_pos + t
            return estimated_pos
        else:
            print("No marker data in previous frame. Falling back to filter prediction.")
            return None

    def _estimate_all_missing(self, missing_marker: str, group_members: list[str]):
        """
        Estimates the missing marker position when all markers in its group are missing
        using the constant rotation assumption from t-2 to t-1.

        Args:
            missing_marker: The name of the missing marker
            group_members: List of marker names in the group
       """
        marker_pos_t_minus_2 = []
        marker_pos_t_minus_1 = []

        for marker_name in group_members + [missing_marker]:
            pos_t_minus_2 = self.second_last_known_positions.get(marker_name)
            pos_t_minus_1 = self.last_known_positions.get(marker_name)

            if pos_t_minus_2 is not None and pos_t_minus_1 is not None:
                marker_pos_t_minus_1.append(pos_t_minus_1)
                marker_pos_t_minus_2.append(pos_t_minus_2)

        if len(marker_pos_t_minus_2) != len(marker_pos_t_minus_1) or \
                len(marker_pos_t_minus_2) < 3 or \
                len(marker_pos_t_minus_1) < 3:
            print(
                f"Incomplete data in frames t-1 and t-2 for {missing_marker} group members. Falling back to filter prediction.")
            return None

        src_pts = np.array(marker_pos_t_minus_2)
        dst_pts = np.array(marker_pos_t_minus_1)

        R, t = self._calculate_rigid_transform(src_pts, dst_pts)

        x_t_minus_1 = self.last_known_positions.get(missing_marker)
        estimated_pos = R @ x_t_minus_1 + t

        return estimated_pos

    def estimate_frame(self, frame_data: dict[str, tuple[tuple, bool]]):
        """
        Estimates missing markers for a single frame.

        Args:
            frame_data: Dictionary {marker_name: ((x, y, z), is_occluded)}
        """
        self.second_last_known_positions = self.last_known_positions.copy()

        final_estimated_positions = {}

        for marker_name in self.marker_groups.keys():
            kf = self.filters.get(marker_name)
            is_occluded = frame_data.get(marker_name)[1]

            if not is_occluded:
                observation = frame_data.get(marker_name)[0]

                new_mean, new_cov = kf.filter_update(kf.initial_state_mean, kf.initial_state_covariance,
                                                     observation=observation)
                kf.initial_state_mean = new_mean
                kf.initial_state_covariance = new_cov

                final_estimated_positions[marker_name] = np.array(observation)
                self.last_known_positions[marker_name] = np.array(observation)

            else:
                estimated_pos = None
                # TODO: W gotowym systemie sam kalman daje rade a przekształcenia z rigid body do porównania
                #  w pracy magisterskiej

                # marker_neighbors = self.marker_groups.get(marker_name)
                # visible_in_group: dict[str, np.ndarray] = {
                #     m: np.array(frame_data.get(m)[0]) for m in marker_neighbors
                #     if not frame_data.get(m)[1]
                # }
                # num_visible_in_group = len(visible_in_group)
                #
                # if num_visible_in_group == 3:
                #     estimated_pos = self._estimate_from_rigid_body(marker_name, visible_in_group)
                #
                # elif num_visible_in_group == 2:
                #     estimated_pos = self._estimate_two_visible(marker_name, visible_in_group)
                #
                # elif num_visible_in_group == 1:
                #     estimated_pos = self._estimate_one_visible(marker_name, visible_in_group)
                #
                # elif num_visible_in_group == 0:
                #     estimated_pos = self._estimate_all_missing(marker_name, marker_neighbors)
                #
                # new_mean, new_cov = kf.filter_update(kf.initial_state_mean, kf.initial_state_covariance,
                #                                      observation=estimated_pos)

                kf.initial_state_mean = new_mean
                kf.initial_state_covariance = new_cov
                final_estimated_positions[marker_name] = new_mean[:self.n_dims]
                self.last_known_positions[marker_name] = new_mean[:self.n_dims]

        return final_estimated_positions
