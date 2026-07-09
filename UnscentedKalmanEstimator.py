import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from numba import njit


@njit(cache=True)
def _calculate_rigid_transform_fast(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Kabsch algorithm for rigid body transformation."""
    src_centroid = np.zeros(3)
    dst_centroid = np.zeros(3)
    for i in range(src_pts.shape[0]):
        src_centroid += src_pts[i]
        dst_centroid += dst_pts[i]
    src_centroid /= src_pts.shape[0]
    dst_centroid /= src_pts.shape[0]

    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        # Reflection correction
        Vt_new = Vt.copy()
        Vt_new[-1, :] *= -1.0
        R = Vt_new.T @ U.T

    t = dst_centroid - R @ src_centroid
    return R, t


@njit(cache=True)
def _find_closest_point_on_circle_fast(circle_center, circle_radius, circle_normal, point):
    v_cc_point = point - circle_center
    offset = np.dot(v_cc_point, circle_normal)
    p_proj = point - offset * circle_normal

    v_cc_proj = p_proj - circle_center
    norm_v = np.linalg.norm(v_cc_proj)

    if norm_v < 1e-9:
        return point
    else:
        return circle_center + circle_radius * (v_cc_proj / norm_v)


class UnscentedKalmanEstimator:
    def __init__(self, marker_groups: dict[str, list[str]], n_dims: int = 3, dt: float = 0.01):
        self.marker_groups = marker_groups
        self.dt = dt
        self.n_dims = n_dims

        # Noise Params
        self.process_noise = 11000.0
        self.observation_noise = 1.0
        self.epsilon = 0.5

        # UKF Params
        self.alpha = 0.1
        self.beta = 2.0
        self.kappa = 0.0

        # Pre-compute Constant Matrices to speed up UKF evaluation
        self._F = self._compute_transition_matrix()
        self._Q = self._compute_process_noise_matrix()
        self._R = np.eye(3) * self.observation_noise * (self.epsilon ** 2)

        self.last_known_positions = {}
        self.second_last_known_positions = {}

        # Initialize Filters
        self.filters = {m: self._create_kalman_filter() for m in marker_groups.keys()}
        self._warmup_numba()

    def _warmup_numba(self):
        dummy_point = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        dummy_pts = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float64)

        _calculate_rigid_transform_fast(dummy_pts, dummy_pts)
        _find_closest_point_on_circle_fast(dummy_point, 1.0, dummy_point, dummy_point)

    def _compute_transition_matrix(self):
        dt = self.dt
        dt2_2 = (dt ** 2) / 2.0
        v_damp, a_damp = 0.98, 0.50
        # 9D state: [x, y, z, vx, vy, vz, ax, ay, az]
        F = np.zeros((9, 9))
        np.fill_diagonal(F, 1.0)
        # Position updates from velocity
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        # Position updates from acceleration
        F[0, 6] = F[1, 7] = F[2, 8] = dt2_2
        # Velocity updates from acceleration
        F[3, 6] = F[4, 7] = F[5, 8] = dt
        # Apply damping
        F[3, 3] = F[4, 4] = F[5, 5] = v_damp
        F[6, 6] = F[7, 7] = F[8, 8] = a_damp
        return F

    def _compute_process_noise_matrix(self):
        dt = self.dt
        q11, q12, q13 = (dt ** 5) / 20.0, (dt ** 4) / 8.0, (dt ** 3) / 6.0
        q22, q23, q33 = (dt ** 3) / 3.0, (dt ** 2) / 2.0, dt

        Q = np.zeros((9, 9))
        for i in range(3):
            idx = [i, i + 3, i + 6]
            block = np.array([[q11, q12, q13],
                              [q12, q22, q23],
                              [q13, q23, q33]])
            for r_idx, row in enumerate(idx):
                for c_idx, col in enumerate(idx):
                    Q[row, col] = block[r_idx, c_idx]
        return Q * self.process_noise

    def _fx(self, state, dt):
        """State transition function utilizing the precomputed F matrix"""
        return self._F @ state

    def _hx(self, state):
        """Observation function slicing the first 3 elements"""
        return state[:3]

    def _create_kalman_filter(self):
        dim_x = 9
        dim_z = 3

        points = MerweScaledSigmaPoints(
            n=dim_x,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa
        )

        kf = UnscentedKalmanFilter(
            dim_x=dim_x,
            dim_z=dim_z,
            dt=self.dt,
            fx=self._fx,
            hx=self._hx,
            points=points
        )

        kf.Q = self._Q
        kf.R = self._R
        kf.x = np.zeros(dim_x)
        kf.P = np.eye(dim_x) * 10.0

        return kf

    def estimate_frame(self, frame_data: dict[str, np.ndarray]):
        self.second_last_known_positions = self.last_known_positions.copy()
        final_positions = {}

        # Categorize visibility
        visible_markers = {m: p for m, p in frame_data.items() if not np.any(np.isnan(p))}

        for marker_name, kf in self.filters.items():
            pos, is_occluded = frame_data[marker_name], np.any(np.isnan(frame_data[marker_name]))

            # Cold start (UKF uses 1D arrays, so we just slice [:3])
            if marker_name not in self.last_known_positions and not is_occluded:
                kf.x[:3] = pos

            kf.predict()

            kf_observation = None
            if not is_occluded:
                kf_observation = pos
            else:
                # Group estimation logic
                neighbors = self.marker_groups.get(marker_name, [])
                visible_in_group = {m: visible_markers[m] for m in neighbors if m in visible_markers}
                num_vis = len(visible_in_group)

                if num_vis >= 3:
                    kf_observation = self._estimate_from_rigid_body(marker_name, visible_in_group)
                elif num_vis == 2:
                    kf_observation = self._estimate_two_visible(marker_name, visible_in_group)
                elif num_vis == 1:
                    kf_observation = self._estimate_one_visible(marker_name, visible_in_group)
                else:
                    kf_observation = self._estimate_all_missing(marker_name, neighbors)

            if kf_observation is not None:
                kf.update(kf_observation)

            final_pos = kf.x[:3]
            final_positions[marker_name] = pos if not is_occluded else final_pos
            self.last_known_positions[marker_name] = final_pos

        return final_positions

    def _estimate_from_rigid_body(self, missing_marker, visible_group):
        missing_prev_pos = self.last_known_positions.get(missing_marker)
        if missing_prev_pos is None:
            return None

        src, dst = [], []
        for m, curr_p in visible_group.items():
            prev_p = self.last_known_positions.get(m)
            if prev_p is not None:
                src.append(prev_p)
                dst.append(curr_p)

        if len(src) < 3:
            return None

        R, t = _calculate_rigid_transform_fast(np.array(src), np.array(dst))
        return R @ missing_prev_pos + t

    def _estimate_two_visible(self, missing_marker, visible_group):
        m2, m3 = list(visible_group.keys())
        x2_t, x3_t = visible_group[m2], visible_group[m3]

        x1_prev = self.last_known_positions.get(missing_marker)
        x2_prev = self.last_known_positions.get(m2)
        x3_prev = self.last_known_positions.get(m3)

        if x1_prev is None or x2_prev is None or x3_prev is None:
            return None

        # Calculate Dt-1 vectors
        D12_t_minus_1 = x2_prev - x1_prev
        D13_t_minus_1 = x3_prev - x1_prev

        # Calculate average position in current frame
        x_mean_t = ((x2_t - D12_t_minus_1) + (x3_t - D13_t_minus_1)) / 2.0

        # Calculate distance between centers and radii
        d = np.linalg.norm(x3_t - x2_t)
        D12_radius = np.linalg.norm(D12_t_minus_1)
        D13_radius = np.linalg.norm(D13_t_minus_1)

        # Check if a circular intersection exists
        if not np.abs(D12_radius - D13_radius) < d < D12_radius + D13_radius:
            return x_mean_t

        # Distance from x2_t to the intersection circle center
        h = (d ** 2 - D13_radius ** 2 + D12_radius ** 2) / (2 * d)

        inter_circle_radius_squared = D12_radius ** 2 - h ** 2
        if inter_circle_radius_squared < 0:
            return x_mean_t

        inter_circle_radius = np.sqrt(inter_circle_radius_squared)
        v_norm = (x3_t - x2_t) / d
        inter_circle_center = x2_t + h * v_norm

        closest_point = _find_closest_point_on_circle_fast(
            inter_circle_center,
            inter_circle_radius,
            v_norm,
            x_mean_t
        )

        return closest_point

    def _estimate_one_visible(self, missing_marker, visible_group):
        m2 = list(visible_group.keys())[0]
        x2_t = visible_group[m2]
        x1_prev, x2_prev = self.last_known_positions.get(missing_marker), self.last_known_positions.get(m2)
        if x1_prev is None or x2_prev is None: return None
        return x2_t - (x2_prev - x1_prev)

    def _estimate_all_missing(self, missing_marker, group):
        src, dst = [], []
        for m in group + [missing_marker]:
            p2, p1 = self.second_last_known_positions.get(m), self.last_known_positions.get(m)
            if p2 is not None and p1 is not None:
                src.append(p2)
                dst.append(p1)
        if len(src) < 3: return None
        R, t = _calculate_rigid_transform_fast(np.array(src), np.array(dst))
        return R @ self.last_known_positions[missing_marker] + t
