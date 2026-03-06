import numpy as np
# import torch.nn as nn
import pandas as pd

# =========================
# LSTM XYZ (OFFLINE) ESTIMATOR
# =========================
# Ten estimator jest przeznaczony do nowego modelu trenowanego bezpośrednio w XYZ:
# - wejście: (XYZ root-centered w metrach) + maska braków
# - wyjście: XYZ root-centered w metrach
# Następnie wracamy do mm i dodajemy root (LASI).
#
# Działa OFFLINE na całym pliku (lub chunkach seq_len), więc daje poprawne wyniki do animacji.


TARGET_MARKERS = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK',
                  'LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB', 'LFIN',
                  'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN',
                  'LASI', 'RASI', 'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE',
                  'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']


class BiLSTM_XYZ(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            in_dim, hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden * 2, out_dim)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.fc(y)


def _compute_missing_mask(df: pd.DataFrame, markers):
    """miss[T,M] True jeśli marker ma NaN w (X lub Y lub Z)"""
    T = len(df)
    M = len(markers)
    miss = np.zeros((T, M), dtype=bool)
    for i, m in enumerate(markers):
        cols = [f"{m}_X", f"{m}_Y", f"{m}_Z"]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        miss[:, i] = df[cols].isna().any(axis=1).to_numpy()
    return miss


def _load_xyz_mm(df: pd.DataFrame, markers):
    """Zwraca xyz[T,M,3] w mm, NaN->0"""
    T = len(df)
    M = len(markers)
    xyz = np.zeros((T, M, 3), dtype=np.float32)

    for m in markers:
        for ax in ["X", "Y", "Z"]:
            col = f"{m}_{ax}"
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for i, m in enumerate(markers):
        xyz[:, i, 0] = df[f"{m}_X"].to_numpy(dtype=np.float32)
        xyz[:, i, 1] = df[f"{m}_Y"].to_numpy(dtype=np.float32)
        xyz[:, i, 2] = df[f"{m}_Z"].to_numpy(dtype=np.float32)

    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    return xyz


def _preprocess_root_centered_m(xyz_mm: np.ndarray, root_idx: int):
    """xyz_mm[T,M,3] -> centered_m[T,M,3] w metrach + root_mm[T,1,3]"""
    xyz_mm = np.nan_to_num(xyz_mm, nan=0.0, posinf=0.0, neginf=0.0)
    root_mm = xyz_mm[:, root_idx:root_idx + 1, :]  # [T,1,3]
    centered_mm = xyz_mm - root_mm
    centered_m = centered_mm / 1000.0  # mm -> m

    centered_m = np.nan_to_num(centered_m, nan=0.0, posinf=0.0, neginf=0.0)
    centered_m = np.clip(centered_m, -3.0, 3.0)
    return centered_m, root_mm


def _predict_full_sequence(model, Xfull_t: torch.Tensor, seq_len: int, device: str, hop: int = 30):
    """
    Overlap-add inference:
      - okna długości seq_len
      - przesuwamy co hop (np. 30)
      - w części wspólnej uśredniamy wagami trójkątnymi
    Xfull_t: [T, in_dim] (CPU tensor)
    Zwraca:  [T, out_dim] (CPU tensor)
    """
    model.eval()
    T = Xfull_t.shape[0]

    # wylicz out_dim przez krótkie forward na sztucznej próbce
    in_dim = Xfull_t.shape[1]
    with torch.no_grad():
        dummy = torch.zeros((1, seq_len, in_dim), dtype=Xfull_t.dtype).to(device)
        out_dim = model(dummy).shape[-1]

    acc = torch.zeros((T, out_dim), dtype=torch.float32)  # suma ważona
    wsum = torch.zeros((T, 1), dtype=torch.float32)  # suma wag

    # wagi trójkątne: małe na brzegach, duże w środku (mniej „szwów”)
    w = torch.linspace(0.0, 1.0, steps=seq_len // 2, dtype=torch.float32)
    if seq_len % 2 == 0:
        w_full = torch.cat([w, w.flip(0)])
    else:
        w_full = torch.cat([w, torch.tensor([1.0]), w.flip(0)])
    w_full = w_full / w_full.max()  # [seq_len], max=1

    with torch.no_grad():
        start = 0
        while start < T:
            end = start + seq_len
            chunk = Xfull_t[start:end]  # [<=seq_len, in_dim]

            if chunk.shape[0] < seq_len:
                pad = seq_len - chunk.shape[0]
                chunk = torch.cat([chunk, torch.zeros((pad, in_dim), dtype=chunk.dtype)], dim=0)

            pred = model(chunk.unsqueeze(0).to(device)).cpu()[0]  # [seq_len, out_dim]

            valid = min(seq_len, T - start)
            ww = w_full[:valid].unsqueeze(1)  # [valid,1]

            acc[start:start + valid] += pred[:valid] * ww
            wsum[start:start + valid] += ww

            start += hop

    # unikamy dzielenia przez 0
    wsum[wsum == 0] = 1.0
    return acc / wsum


def blend_gap_boundaries(filled_mm, observed_mm, miss, k=5):
    """
    Wygładza wejście/wyjście z dziury.
    filled_mm: [T,M,3] po LSTM
    observed_mm: oryginalne dane z GAP (NaN->0)
    miss: [T,M] maska braków
    k: liczba klatek do blendowania
    """
    T, M, _ = filled_mm.shape
    out = filled_mm.copy()

    for m in range(M):
        mask = miss[:, m]
        t = 0
        while t < T:
            if not mask[t]:
                t += 1
                continue

            start = t
            while t < T and mask[t]:
                t += 1
            end = t  # zakres [start, end)

            # blend początek
            if start - 1 >= 0 and not mask[start - 1]:
                for i in range(1, k + 1):
                    tt = start + (i - 1)
                    if tt >= end:
                        break
                    alpha = i / (k + 1)
                    out[tt, m, :] = (
                            (1 - alpha) * observed_mm[start - 1, m, :]
                            + alpha * out[tt, m, :]
                    )

            # blend koniec
            if end < T and not mask[end]:
                for i in range(1, k + 1):
                    tt = end - i
                    if tt < start:
                        break
                    alpha = i / (k + 1)
                    out[tt, m, :] = (
                            (1 - alpha) * observed_mm[end, m, :]
                            + alpha * out[tt, m, :]
                    )

    return out


def build_edges_from_marker_groups(marker_groups):
    edges = set()
    for a, neigh in marker_groups.items():
        for b in neigh:
            if a == b:
                continue
            edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def compute_reference_bone_lengths_from_sequence(xyz_mm, edges, marker_to_idx):
    """
    xyz_mm: [T,M,3] (filled albo gap_mm)
    Zwraca dict[(m1,m2)] = median length (mm) z całej sekwencji.
    """
    ref = {}
    for (m1, m2) in edges:
        i1 = marker_to_idx.get(m1)
        i2 = marker_to_idx.get(m2)
        if i1 is None or i2 is None:
            continue
        d = xyz_mm[:, i2, :] - xyz_mm[:, i1, :]
        dist = np.linalg.norm(d, axis=1)
        dist = dist[np.isfinite(dist)]
        if len(dist) == 0:
            continue
        ref[(m1, m2)] = float(np.median(dist))
    return ref


def bone_error_per_frame(xyz_mm, edges, marker_to_idx, ref_lengths):
    """
    Zwraca err[t] = średni względny błąd długości kości w klatce t.
    """
    T = xyz_mm.shape[0]
    err = np.zeros((T,), dtype=np.float32)

    for t in range(T):
        e_sum = 0.0
        e_cnt = 0
        for (m1, m2) in edges:
            L = ref_lengths.get((m1, m2))
            if L is None or L < 1e-6:
                continue
            i1 = marker_to_idx.get(m1)
            i2 = marker_to_idx.get(m2)
            if i1 is None or i2 is None:
                continue
            dist = float(np.linalg.norm(xyz_mm[t, i2, :] - xyz_mm[t, i1, :]))
            if not np.isfinite(dist):
                continue
            e_sum += abs(dist - L) / L  # względny błąd
            e_cnt += 1
        err[t] = (e_sum / e_cnt) if e_cnt > 0 else 0.0

    return err


def detect_bad_frames_by_bone_error(err, z=6.0, min_err=0.06):
    """
    err: [T] (średni względny błąd długości kości)
    Zwraca maskę bad frame.
    - z: ile odchyleń standardowych ponad średnią uznajemy za outlier
    - min_err: minimalny absolutny próg (np. 0.06 = 6%)
    """
    mu = float(np.mean(err))
    sd = float(np.std(err) + 1e-8)
    thr = max(mu + z * sd, min_err)
    bad = err > thr
    return bad, thr, mu, sd


def fix_bad_frames_interpolate_missing_only(xyz_mm, miss, bad_frames):
    """
    xyz_mm: [T,M,3] (filled)
    miss:   [T,M] (True = marker był missing w GAP)
    bad_frames: [T] bool (outlier klatki)
    Naprawiamy tylko:
      - klatki bad_frames
      - i tylko markery, które były missing
    Interpolacja liniowa między najbliższymi dobrymi klatkami.
    """
    out = xyz_mm.copy()
    T, M, _ = out.shape

    good_idx = np.where(~bad_frames)[0]
    if len(good_idx) < 2:
        return out  # nic nie zrobimy

    for t in np.where(bad_frames)[0]:
        # znajdź najbliższą dobrą klatkę z lewej i prawej
        left = good_idx[good_idx < t]
        right = good_idx[good_idx > t]
        if len(left) == 0 or len(right) == 0:
            continue
        tl = left[-1]
        tr = right[0]
        if tr == tl:
            continue
        alpha = (t - tl) / (tr - tl)

        for m in range(M):
            if not miss[t, m]:
                continue
            out[t, m, :] = (1 - alpha) * out[tl, m, :] + alpha * out[tr, m, :]

    return out


def clamp_velocity_missing_only_per_marker(xyz_mm, miss, markers, default_max=50.0, special=None):
    """
    Ogranicza skok mm/frame, tylko jeśli miss[t,m]==True.
    special: dict marker_name -> max_jump_mm
    """
    if special is None:
        special = {}

    out = xyz_mm.copy()
    T, M, _ = out.shape
    for t in range(1, T):
        for mi in range(M):
            if not miss[t, mi]:
                continue
            mname = markers[mi]
            max_jump = float(special.get(mname, default_max))

            prev = out[t - 1, mi, :]
            cur = out[t, mi, :]
            d = cur - prev
            dist = float(np.linalg.norm(d))
            if dist > max_jump and dist > 1e-6:
                out[t, mi, :] = prev + d * (max_jump / dist)
    return out


class LSTMXYZEstimator:
    """OFFLINE: wypełnia dziury w całym pliku CSV naraz (chunkami seq_len)."""

    def __init__(self, model_path, root_marker="LASI", seq_len=60):
        self.root_marker = root_marker
        self.seq_len = seq_len
        self.markers = TARGET_MARKERS
        self.root_idx = self.markers.index(self.root_marker)

        # Stabilizacja: u części konfiguracji GPU/cuDNN LSTM potrafi dawać NaN
        torch.backends.cudnn.enabled = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        in_dim = len(self.markers) * 4
        out_dim = len(self.markers) * 3

        self.model = BiLSTM_XYZ(in_dim, out_dim, hidden=256, layers=2, dropout=0.1).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[LSTM-XYZ] Loaded checkpoint: {model_path}")
        print(f"[LSTM-XYZ] Model params: {n_params:,} | hidden=256 layers=2 dropout=0.1 | seq_len={self.seq_len}")
        print(f"[LSTM-XYZ] device={self.device} | cuDNN={torch.backends.cudnn.enabled}")

        print(f"[LSTM-XYZ] Model loaded: {model_path} | device={self.device} | cuDNN={torch.backends.cudnn.enabled}")

    @staticmethod
    def stabilize_start_missing(xyz_mm, miss, n_warmup=30):
        """
        Jeśli na początku sekwencji marker jest missing, to zapobiega odlotom:
        - znajduje pierwszą klatkę, gdzie marker nie jest missing
        - kopiuje tę pozycję wstecz na początek (tylko dla klatek missing)
        """
        out = xyz_mm.copy()
        T, M, _ = out.shape
        n = min(n_warmup, T)

        for m in range(M):
            if not miss[:n, m].any():
                continue
            good = np.where(~miss[:, m])[0]
            if len(good) == 0:
                continue
            first_good = good[0]
            anchor = out[first_good, m, :].copy()

            for t in range(n):
                if miss[t, m]:
                    out[t, m, :] = anchor

        return out

    def estimate_dataframe(self, df_gap: pd.DataFrame) -> pd.DataFrame:
        df = df_gap.copy()

        miss = _compute_missing_mask(df.copy(), self.markers)  # [T,M]
        gap_mm = _load_xyz_mm(df, self.markers)  # [T,M,3] mm (NaN->0)

        gap_m, root_mm = _preprocess_root_centered_m(gap_mm, self.root_idx)  # [T,M,3] m

        Xfull = np.concatenate([gap_m, miss.astype(np.float32)[..., None]], axis=2)  # [T,M,4]
        Xfull = Xfull.reshape(len(df), -1).astype(np.float32)  # [T,in_dim]

        Xfull_t = torch.tensor(Xfull, dtype=torch.float32)
        Ypred_t = _predict_full_sequence(self.model, Xfull_t, self.seq_len, self.device, hop=30)

        Ypred = Ypred_t.numpy().reshape(len(df), len(self.markers), 3)  # m
        Ypred_mm = (Ypred * 1000.0) + root_mm  # mm

        filled = gap_mm.copy()
        for i in range(len(self.markers)):
            idx = miss[:, i]
            filled[idx, i, :] = Ypred_mm[idx, i, :]

        # 0) Stabilizacja początku (ważne!)
        filled = self.stabilize_start_missing(filled, miss, n_warmup=30)

        # 1) Clamp tylko dla missing
        special_limits = {
            "RTOE": 30.0,
            "LTOE": 30.0,
            "RFIN": 35.0,
            "LFRM": 40.0,
            "RTHI": 40.0,
        }
        filled = clamp_velocity_missing_only_per_marker(
            filled, miss, self.markers,
            default_max=50.0,
            special=special_limits
        )

        # 2) Blend granic dziur (TYLKO RAZ)
        filled = blend_gap_boundaries(filled, gap_mm, miss, k=5)

        # 3) Outlier frames po kościach + interpolacja (zostaw)
        from marker_groups import MARKER_GROUPS
        edges = build_edges_from_marker_groups(MARKER_GROUPS)
        marker_to_idx = {m: i for i, m in enumerate(self.markers)}
        ref_lengths = compute_reference_bone_lengths_from_sequence(filled, edges, marker_to_idx)
        err = bone_error_per_frame(filled, edges, marker_to_idx, ref_lengths)
        bad_frames, thr, mu, sd = detect_bad_frames_by_bone_error(err, z=6.0, min_err=0.06)
        print(f"[LSTM-XYZ] bone-error outliers: {int(bad_frames.sum())} frames | thr={thr:.4f} mu={mu:.4f} sd={sd:.4f}")
        filled = fix_bad_frames_interpolate_missing_only(filled, miss, bad_frames)

        # wpisz do DataFrame (kolumny jak w wejściu)
        for i, m in enumerate(self.markers):
            df[f"{m}_X"] = filled[:, i, 0]
            df[f"{m}_Y"] = filled[:, i, 1]
            df[f"{m}_Z"] = filled[:, i, 2]

        return df


def clamp_velocity_missing_only(xyz_mm, miss, max_jump_mm=120.0):
    """
    Ogranicza skok pozycji między klatkami do max_jump_mm
    TYLKO dla markerów, które były missing w tej klatce.
    xyz_mm: [T,M,3]
    miss:   [T,M] (True = było missing w wejściu GAP)
    """
    out = xyz_mm.copy()
    T, M, _ = out.shape

    for t in range(1, T):
        for m in range(M):
            if not miss[t, m]:
                continue  # nie ruszamy obserwacji
            prev = out[t - 1, m, :]
            cur = out[t, m, :]
            d = cur - prev
            dist = float(np.linalg.norm(d))
            if dist > max_jump_mm and dist > 1e-6:
                out[t, m, :] = prev + d * (max_jump_mm / dist)

    return out
