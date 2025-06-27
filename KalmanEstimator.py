from pykalman import KalmanFilter
import numpy as np


class KalmanEstimator:
    def __init__(self, marker_groups, n_dims=3, dt=1 / 100, process_noise=0.001, observation_noise=0.01):
        """
        Args:
            marker_groups: Dictionary mapping marker names to their neighbor groups
            dt: Time step (default 100 Hz sampling rate)
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

        # Process noise covariance
        Q = np.eye(6) * self.process_noise

        # Observation noise covariance
        R = np.eye(3) * self.observation_noise

        # Initial state covariance
        P0 = np.eye(6)

        kf = KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            initial_state_mean=np.zeros(self.n_dims * 2),
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_covariance=P0
        )

        return kf

    @staticmethod
    def _estimate_from_mean(visible_markers_data):
        """
        Estimates the missing marker position by taking the mean of visible markers in the group.
        """
        return np.mean(np.array(visible_markers_data.values()), axis=0)

    @staticmethod
    def _find_closest_point_on_circle(circle_center, circle_radius, circle_normal, point):
        """
        Finds the closest point on a 3D circle to a given point
        """
        # Project point onto radical plane
        v_cc_point = point - circle_center
        offset = v_cc_point @ circle_normal
        v_proj = point - offset * circle_normal

        # Find the closest point on the circle
        v_cc_proj = v_proj - circle_center

        if np.isclose(np.linalg.norm(v_cc_proj), 0):
            # Projected point lies in a circle center

            # Get not parallel vector to the circle normal
            vec = np.array([1.0, 0.0, 0.0])
            if np.isclose(np.linalg.norm(np.cross(circle_normal, vec)), 0):
                vec = np.array([0.0, 1.0, 0.0])

            # Get a perpendicular vector to circle normal
            u_vec = np.cross(circle_normal, vec)
            u_vec_norm = u_vec / np.linalg.norm(u_vec)

            closest_point = circle_center + circle_radius * u_vec_norm
        else:
            # The closest point lies on a path from circle center to projected point

            v_cc_proj_norm = v_cc_proj / np.linalg.norm(v_cc_proj)
            closest_point = circle_center + circle_radius * v_cc_proj_norm

        return closest_point

    def _estimate_two_visible(self, missing_marker, visible_markers_data):
        """
        Estimates the missing marker position when exactly two markers in its group are visible

        Args:
            missing_marker: The name of the missing marker.
            visible_markers_data: Dictionary of the two visible markers in the group
                                         and their current positions.
        """
        if len(visible_markers_data) != 2:
            raise ValueError("This function requires exactly two visible markers in the group.")

        m1 = missing_marker
        m2, m3 = list(visible_markers_data.keys())
        x2_t, x3_t = visible_markers_data.get(m2), visible_markers_data.get(m3)

        # Retrieve previous frame's estimated positions (x_t-1)
        x1_t_minus_1 = self.last_known_positions.get(m1)
        x2_t_minus_1 = self.last_known_positions.get(m2)
        x3_t_minus_1 = self.last_known_positions.get(m3)

        if x1_t_minus_1 is None or x2_t_minus_1 is None or x3_t_minus_1 is None:
            print(f"Warning: Previous frame data incomplete for {m1}, {m2}, {m3}",
                  f"Falling back to estimation from mean.")
            return self._estimate_from_mean(visible_markers_data)

        # Calculate Dt-1 vectors
        D12_t_minus_1 = x2_t_minus_1 - x1_t_minus_1
        D13_t_minus_1 = x3_t_minus_1 - x1_t_minus_1

        # Calculate average position in current frame
        x_mean_t = ((x2_t - D12_t_minus_1) + (x3_t - D13_t_minus_1)) / 2.0

        # Calculate the distance between 2 spheres with centers at x2_t and x3_t
        # and radius |D12_t_minus_1| and |D13_t_minus_1| respectively
        d = np.linalg.norm(x3_t - x2_t)

        # Calculate radius values for spheres
        D12 = np.linalg.norm(D12_t_minus_1)
        D13 = np.linalg.norm(D13_t_minus_1)

        if not np.abs(D12 - D13) < d < D12 + D13:
            # No circular intersection
            return x_mean_t

        # Calculate the distance between a sphere center at x2_t and the intersection circle center
        h = (d ** 2 - D13 ** 2 + D12 ** 2) / (2 * d)
        rc_squared = D12 ** 2 - h ** 2

        if rc_squared < 0:
            # No real intersection
            return x_mean_t

        # Calculate intersection circle radius
        rc = np.sqrt(rc_squared)
        # Calculate intersection circle center
        cc = x2_t + h * ((x3_t - x2_t) / d)
        # Calculate normal vector
        n_norm = (x3_t - x2_t) / d

        closest_point = self._find_closest_point_on_circle(cc, rc, n_norm, x_mean_t)

        return closest_point

    def _estimate_one_visible(self, missing_marker, visible_markers_data):
        """
           Estimates the missing marker position when exactly one markers in its group are visible

           Args:
               missing_marker: The name of the missing marker.
               visible_markers_data: Dictionary of the one visible marker in the group
                                            and its current positions.
        """
        if len(visible_markers_data) != 1:
            raise ValueError("This function requires exactly one visible markers in the group.")

        m1 = missing_marker
        m2 = list(visible_markers_data.keys())[0]
        x2_t = visible_markers_data[m2]

        # Retrieve previous frame's positions (x_t-1)
        x1_t_minus_1 = self.last_known_positions.get(m1)
        x2_t_minus_1 = self.last_known_positions.get(m2)

        if x1_t_minus_1 is None or x2_t_minus_1 is None:
            # Cannot perform this estimation without previous frame's data for all 3 markers
            # print(
            #     f"Warning: Previous frame data incomplete for {m1_name}, {m2_name}, {m3_name}. Falling back to simpler estimation (mean).")
            return np.mean(list(visible_markers_data.values()), axis=0)

        D12_t_minus_1 = x2_t_minus_1 - x1_t_minus_1

        x_mean_t = x2_t - D12_t_minus_1

        return x_mean_t

    @staticmethod
    def _compute_rigid_transform(src_pts, dst_pts):
        """
        Computes the optimal rigid body transformation
        Args:
            src_pts (np.array): Nx3 array of source points
            dst_pts (np.array): Nx3 array of destination points
        """
        # if src_pts.shape[0] < self.n_dims or src_pts.shape != dst_pts.shape:
        #     # Need at least 3 points for a 3D rigid transform
        #     # And source/destination point sets must have the same shape
        #     return None, None

        # Center the point clouds
        src_centroid = np.mean(src_pts, axis=0)
        dst_centroid = np.mean(dst_pts, axis=0)

        src_centered = src_pts - src_centroid
        dst_centered = dst_pts - dst_centroid

        # Compute the covariance matrix H
        H = src_centered.T @ dst_centered

        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)

        # Compute the rotation matrix R
        R = Vt.T @ U.T

        # Handle reflection case (if determinant is -1). This ensures a pure rotation.
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute the translation vector t
        t = dst_centroid - R @ src_centroid

        return R, t

    def _estimate_from_rigid_body(self, missing_marker, visible_markers_data):
        """
        Estimates the missing marker position based on the rigid body transformation

        Args:
            missing_marker: The name of the missing marker.
            visible_markers_data: Dictionary of visible markers in the group and their current positions.
        """

        # reference_group_markers_pos = {}

        # # Add the missing marker itself to the reference set if its last known position exists
        # if self.last_known_positions.get(missing_marker) is not None:
        #     reference_group_markers_pos[missing_marker] = self.last_known_positions.get(missing_marker)
        #
        # # Add all group members from to the reference set if they have last known positions
        # for member_marker in self.marker_groups.get(missing_marker):
        #     if self.last_known_positions.get(member_marker) is not None:
        #         reference_group_markers_pos[member_marker] = self.last_known_positions.get(member_marker)

        src_pts = []  # Reference points
        dst_pts = []  # Visible points from current frame

        for marker_name, current_pos in visible_markers_data.items():
            if self.last_known_positions.get(marker_name) is not None:
                src_pts.append(self.last_known_positions.get(marker_name))
                dst_pts.append(current_pos)

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)

        # if len(src_pts) < 3:
        #     # Not enough corresponding points for robust rigid body transformation
        #     return self._estimate_from_mean(visible_markers_data)

        R, t = self._compute_rigid_transform(src_pts, dst_pts)

        # Transform the reference position of the missing marker
        if self.last_known_positions.get(missing_marker) is not None:
            missing_marker_ref_pos = self.last_known_positions.get(missing_marker)
            estimated_pos = R @ missing_marker_ref_pos + t
            return estimated_pos
        else:
            print("Raczej tu nie wejdzie")
            return None
            # return self._estimate_from_mean(visible_markers_data)

    def _estimate_all_missing(self, missing_marker, group_members):
        """
        Estimates the missing marker position when ALL markers in its group are missing
        in the current frame, using the constant rotation assumption from t-2 to t-1.

        Args:
           missing_marker: The name of the missing marker.
           group_members: List of marker names belonging to the missing_marker's group.
       """
        marker_pos_t_minus_2 = {}
        marker_pos_t_minus_1 = {}

        for marker_name in group_members + [missing_marker]:
            pos_t_minus_2 = self.second_last_known_positions.get(marker_name)
            pos_t_minus_1 = self.last_known_positions.get(marker_name)

            if pos_t_minus_2 is not None and pos_t_minus_1 is not None:
                marker_pos_t_minus_1[marker_name] = pos_t_minus_1
                marker_pos_t_minus_2[marker_name] = pos_t_minus_2

        if len(marker_pos_t_minus_2) != len(marker_pos_t_minus_1) or \
                len(marker_pos_t_minus_2) < self.n_dims or \
                len(marker_pos_t_minus_1) < self.n_dims:
            print(
                f"Warning: Not enough history points for group of {missing_marker} for constant rotation. Falling back to KF prediction.")
            return None

        src_pts = np.array(list(marker_pos_t_minus_2.values()))
        dst_pts = np.array(list(marker_pos_t_minus_1.values()))

        R, _ = self._compute_rigid_transform(src_pts, dst_pts)

        x_t_minus_1 = self.last_known_positions.get(missing_marker)
        estimated_pos = R @ x_t_minus_1

        return estimated_pos

    def estimate_frame(self, frame_data):
        """
        Estimate missing markers for a single frame.

        Args:
            frame_data: Dictionary {marker_name: np.array([x, y, z], is_occluded), ...}
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
                marker_neighbors = self.marker_groups.get(marker_name)
                visible_in_group = {
                    m: np.array(frame_data.get(m)[0]) for m in marker_neighbors
                    if not frame_data.get(m)[1]
                }
                num_visible_in_group = len(visible_in_group)

                if num_visible_in_group == 3:
                    estimated_pos = self._estimate_from_rigid_body(marker_name, visible_in_group)

                elif num_visible_in_group == 2:
                    estimated_pos = self._estimate_two_visible(marker_name, visible_in_group)

                elif num_visible_in_group == 1:
                    estimated_pos = self._estimate_one_visible(marker_name, visible_in_group)

                elif num_visible_in_group == 0:
                    estimated_pos = self._estimate_all_missing(marker_name, marker_neighbors)

                new_mean, new_cov = kf.filter_update(kf.initial_state_mean, kf.initial_state_covariance,
                                                     observation=estimated_pos)

                kf.initial_state_mean = new_mean
                kf.initial_state_covariance = new_cov
                final_estimated_positions[marker_name] = new_mean[:self.n_dims]
                self.last_known_positions[marker_name] = new_mean[:self.n_dims]

        return final_estimated_positions
