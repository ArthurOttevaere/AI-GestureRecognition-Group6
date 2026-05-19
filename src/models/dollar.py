import numpy as np
from numba import njit

from src.config import DOLLAR_N, DOLLAR_L, DOLLAR_EPSILON, KNN_K


# ==============================================================================
# 4b.  $1 RECOGNIZER -- 3D ADAPTATION FOLLOWING KRATZ & ROHS (2010)
#     [Advanced method -- placed here for implementation proximity to baseline
#      distance functions; classified as advanced per module docstring Phase 4]
# ==============================================================================
# The 2D $1 Recognizer of Wobbrock, Wilson & Li (2007) is extended to 3D
# according to the canonical procedure of Kratz & Rohs (2010), "A $3
# Gesture Recognizer", Proc. IUI '10, pp. 341-344.
#
# Pipeline
# --------
#   1. Resample to N=64 equidistant 3D points (linear interpolation along
#      cumulative arc length).
#   2. Rotate so that the first resampled point lies along the centroid
#      direction.  The rotation axis is the unit vector pâ x c (cross
#      product); the angle is acos((pâ . c) / (||pâ|| ||c||)).  Rotation
#      applied with Rodrigues' formula.  Degenerate case (pâ collinear
#      with c) -> identity rotation.
#      NOTE: rotation must be applied BEFORE translation so that both p0
#      and c are non-zero (centroid not yet at origin).
#   3. Translate centroid to the origin (after rotation).
#   4. Uniformly rescale so that the longest bounding-box edge equals
#      DOLLAR_L (=1.0). This is the "normalised cube of side l" of
#      Kratz & Rohs (2010).  No axis-by-axis scaling: this avoids the
#      division-by-zero issue on quasi-planar gestures.
#
# Score
# -----
#   S = 1 - d / (0.5 * sqrt(3) * l^2)
# where d is the MSE (mean squared point-to-point distance) between the
# preprocessed candidate and the preprocessed template (Kratz & Rohs,
# 2010).  The GSS (Golden Section Search) further refines alignment on
# the 3 axes before computing d.
#
# Templates are preprocessed only once and cached, as required by
# Wobbrock et al. (2007, "For gestures serving as templates, Steps 1-3
# should be carried out once on the raw input points.").
# ==============================================================================

def _dollar_path_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def dollar_resample(points: np.ndarray, n: int = DOLLAR_N) -> np.ndarray:
    """Step 1 of Kratz & Rohs (2010): resample to n equidistant 3D points."""
    total = _dollar_path_length(points)
    if total == 0.0 or len(points) < 2:
        return np.tile(points[0], (n, 1))

    interval   = total / (n - 1)
    D          = 0.0
    new_points = [points[0].copy()]
    pts        = points.copy()

    i = 1
    while i < len(pts) and len(new_points) < n:
        d = float(np.linalg.norm(pts[i] - pts[i - 1]))
        if D + d >= interval:
            frac = (interval - D) / d
            q    = pts[i - 1] + frac * (pts[i] - pts[i - 1])
            new_points.append(q)
            pts = np.insert(pts, i, q, axis=0)
            D = 0.0
        else:
            D += d
        i += 1

    while len(new_points) < n:
        new_points.append(pts[-1].copy())

    return np.array(new_points[:n], dtype=float)


def _dollar_centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def _dollar_translate_to_origin(points: np.ndarray) -> np.ndarray:
    return points - _dollar_centroid(points)


def _rodrigues_rotate(points: np.ndarray,
                       axis: np.ndarray,
                       angle: float) -> np.ndarray:
    """
    Rotate a (T, 3) array of points around a unit axis by `angle` radians,
    using Rodrigues' rotation formula (Kratz & Rohs, 2010).
        v_rot = v cos(t) + (k x v) sin(t) + k (k . v)(1 - cos(t))
    """
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    kx, ky, kz = axis
    K = np.array([[ 0.0, -kz,  ky],
                  [  kz, 0.0, -kx],
                  [ -ky,  kx, 0.0]])
    R = np.eye(3) * c + K * s + np.outer(axis, axis) * one_minus_c
    return points @ R.T


def _dollar_align_to_indicative_axis(points: np.ndarray) -> np.ndarray:
    """
    Step 3 of Kratz & Rohs (2010): rotate so that the first point pâ aligns
    with the centroid direction. Translation to origin must be applied
    first.
    The rotation axis is pâ x c (unit vector); the angle is the arccos of
    the normalised dot product. Degenerate cases (pâ or c with zero norm,
    or pâ collinear with c) -> identity rotation.
    """
    if len(points) == 0:
        return points
    p1 = points[0]
    c  = _dollar_centroid(points)

    n_p1 = float(np.linalg.norm(p1))
    n_c  = float(np.linalg.norm(c))
    if n_p1 < 1e-12 or n_c < 1e-12:
        return points

    cos_theta = float(np.dot(p1, c) / (n_p1 * n_c))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta     = float(np.arccos(cos_theta))

    cross   = np.cross(p1, c)
    n_cross = float(np.linalg.norm(cross))
    if n_cross < 1e-12 or theta < 1e-9:
        # Degenerate: pâ already collinear with c -> identity.
        return points
    axis = cross / n_cross
    return _rodrigues_rotate(points, axis, theta)


def _dollar_scale_cube(points: np.ndarray,
                        l: float = DOLLAR_L) -> np.ndarray:
    """
    Step 4 (Kratz & Rohs, 2010): uniform rescaling INSIDE a normalised
    cube of side l.  The longest bounding-box edge becomes l. Avoids the
    division-by-zero issue of axis-by-axis scaling on quasi-planar
    gestures.
    """
    extents = points.max(axis=0) - points.min(axis=0)
    max_ext = float(extents.max())
    if max_ext < 1e-12:
        return points
    return points * (l / max_ext)


# -- Golden Section Search (GSS) helpers — Numba JIT --------------------------
# Kratz & Rohs (2010), "Search for Minimum Distance at Best Angle":
#   phi = 0.5*(sqrt(5)-1), angular range [-180°,180°], cutoff = 2°, 11 iters.
# Three functions:
#   _rotate_1axis_nb : rotate (N,3) array around one principal axis
#   _mse_nb          : mean squared Euclidean distance between two (N,3) arrays
#   _dollar_gss_mse  : sequential 3-axis GSS returning the minimum MSE
# All are @njit(cache=True) for maximum speed (called 10^4+ times in CV).

@njit(cache=True)
def _rotate_1axis_nb(pts: np.ndarray, axis_idx: int,
                     theta: float) -> np.ndarray:
    """Rotate (N,3) array around principal axis axis_idx by theta radians."""
    n   = pts.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    c   = np.cos(theta)
    s   = np.sin(theta)
    for i in range(n):
        x = pts[i, 0]
        y = pts[i, 1]
        z = pts[i, 2]
        if axis_idx == 0:           # x-axis
            out[i, 0] = x
            out[i, 1] = y * c - z * s
            out[i, 2] = y * s + z * c
        elif axis_idx == 1:         # y-axis
            out[i, 0] =  x * c + z * s
            out[i, 1] =  y
            out[i, 2] = -x * s + z * c
        else:                       # z-axis
            out[i, 0] = x * c - y * s
            out[i, 1] = x * s + y * c
            out[i, 2] = z
    return out


@njit(cache=True)
def _mse_nb(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared point-to-point distance between two (N,3) arrays."""
    n     = a.shape[0]
    total = 0.0
    for i in range(n):
        dx = a[i, 0] - b[i, 0]
        dy = a[i, 1] - b[i, 1]
        dz = a[i, 2] - b[i, 2]
        total += dx * dx + dy * dy + dz * dz
    return total / n


@njit(cache=True)
def _dollar_gss_mse(cand: np.ndarray, tmpl: np.ndarray) -> float:
    """
    Golden Section Search over 3 rotation axes (Kratz & Rohs 2010).
    Applies GSS sequentially on x, y, z axes; rotates candidate to best
    angle on each axis before moving to the next.
    Parameters: phi = 0.5*(sqrt(5)-1), cutoff = 2 deg = pi/90, 11 iters.
    Returns the minimum MSE after optimal 3-axis alignment.
    """
    phi    = 0.5 * (-1.0 + np.sqrt(5.0))
    cutoff = np.pi / 90.0       # 2 degrees
    n      = cand.shape[0]

    for axis in range(3):
        a_ang = -np.pi
        b_ang =  np.pi
        for _ in range(11):
            if (b_ang - a_ang) <= cutoff:
                break
            c1 = b_ang - phi * (b_ang - a_ang)
            c2 = a_ang + phi * (b_ang - a_ang)

            # Compute d1 (MSE at c1) and d2 (MSE at c2) inline — no alloc
            cos1 = np.cos(c1);  sin1 = np.sin(c1)
            cos2 = np.cos(c2);  sin2 = np.sin(c2)
            d1 = 0.0;  d2 = 0.0
            for i in range(n):
                x = cand[i, 0];  y = cand[i, 1];  z = cand[i, 2]
                if axis == 0:
                    rx1 = x;  ry1 = y*cos1 - z*sin1;  rz1 = y*sin1 + z*cos1
                    rx2 = x;  ry2 = y*cos2 - z*sin2;  rz2 = y*sin2 + z*cos2
                elif axis == 1:
                    rx1 =  x*cos1 + z*sin1;  ry1 = y;  rz1 = -x*sin1 + z*cos1
                    rx2 =  x*cos2 + z*sin2;  ry2 = y;  rz2 = -x*sin2 + z*cos2
                else:
                    rx1 = x*cos1 - y*sin1;  ry1 = x*sin1 + y*cos1;  rz1 = z
                    rx2 = x*cos2 - y*sin2;  ry2 = x*sin2 + y*cos2;  rz2 = z
                ex1 = rx1 - tmpl[i, 0];  ey1 = ry1 - tmpl[i, 1]
                ez1 = rz1 - tmpl[i, 2]
                ex2 = rx2 - tmpl[i, 0];  ey2 = ry2 - tmpl[i, 1]
                ez2 = rz2 - tmpl[i, 2]
                d1 += ex1*ex1 + ey1*ey1 + ez1*ez1
                d2 += ex2*ex2 + ey2*ey2 + ez2*ez2

            if d1 < d2:
                b_ang = c2
            else:
                a_ang = c1

        best_angle = (a_ang + b_ang) * 0.5
        cand = _rotate_1axis_nb(cand, axis, best_angle)

    return _mse_nb(cand, tmpl)


def dollar_preprocess(points: np.ndarray,
                       n: int   = DOLLAR_N,
                       l: float = DOLLAR_L) -> np.ndarray:
    """
    Apply Steps 1-4 of the $3 (Kratz & Rohs, 2010) preprocessing.
    Used once on training templates (cached) and once on each candidate.
    Correct step order (Kratz & Rohs 2010):
      1. Resample to N equidistant points.
      2. Rotate to indicative angle (p0 and centroid c are both non-zero
         here — centroid has NOT yet been moved to origin).
      3. Translate centroid to origin (after rotation so c stays valid).
      4. Scale uniformly to unit cube of side l.
    NOTE: translation must come AFTER rotation. If translation is applied
    first, the centroid becomes (0,0,0) and the indicative-angle formula
    n_c < 1e-12 guard triggers, making the rotation a no-op.
    """
    pts = dollar_resample(points, n)
    pts = _dollar_align_to_indicative_axis(pts)   # step 2: c valid here
    pts = _dollar_translate_to_origin(pts)        # step 3: centroid → 0
    pts = _dollar_scale_cube(pts, l)              # step 4
    return pts


def _dollar_path_distance(a: np.ndarray, b: np.ndarray) -> float:
    """MSE (mean squared point-to-point distance) between two same-length 3D paths.
    Required by Kratz & Rohs (2010) score formula: S = 1 - d/(0.5*sqrt(3)*l^2).
    """
    return float(np.mean(np.sum((a - b) ** 2, axis=1)))


def dollar_score(distance: float, l: float = DOLLAR_L) -> float:
    """
    Confidence score in [0, 1] (Kratz & Rohs, 2010, 3D MSE adaptation):
        S = 1 - d / (0.5 * sqrt(3) * l^2)
    where d is the MSE between the two preprocessed paths.
    Clipped to [0, 1] in case of numerical edge effects.
    """
    s = 1.0 - distance / (0.5 * np.sqrt(3.0) * l ** 2)
    return float(max(0.0, min(1.0, s)))


def dollar_recognize(candidate_pre: np.ndarray,
                      templates_pre: list,
                      template_labels: list,
                      l: float = DOLLAR_L,
                      allow_rejection: bool = False,
                      epsilon: float = DOLLAR_EPSILON
                      ) -> tuple[int, float, list]:
    """
    Step 5 of Kratz & Rohs (2010) recognition: rank all (preprocessed)
    templates against the (preprocessed) candidate by GSS-refined MSE,
    and return:
        (best_label, best_score, ranked_list)
    where ranked_list is a sorted N-best list:
        [(label, distance, score), ...]   sorted by distance ascending.

    allow_rejection=False (default):
        Always return the best-scoring candidate (1-Best, no rejection).
        Use this in cross-validation so the $3 is directly comparable to
        RF/LR/DT on accuracy (force prediction on every sample).

    allow_rejection=True:
        Apply the Kratz & Rohs (2010) scoring heuristic:
          - If best_score > 1.1*epsilon → return best candidate.
          - Elif within top-3, two entries of the same class both have
            score > 0.95*epsilon → return that class.
          - Else → return (-1, 0.0, ranked) meaning "not recognized".
    """
    distances = [
        _dollar_gss_mse(candidate_pre, t) for t in templates_pre
    ]
    order   = np.argsort(distances)
    ranked  = [(template_labels[k], float(distances[k]),
                dollar_score(distances[k], l)) for k in order]
    best_label, _best_d, best_s = ranked[0]

    if not allow_rejection:
        return int(best_label), float(best_s), ranked

    # --- Kratz & Rohs (2010) scoring heuristic ---
    if best_s > 1.1 * epsilon:
        return int(best_label), float(best_s), ranked

    top3 = ranked[:3]
    counts: dict = {}
    for lbl, _d, sc in top3:
        if sc > 0.95 * epsilon:
            counts[lbl] = counts.get(lbl, 0) + 1
    for lbl, cnt in counts.items():
        if cnt >= 2:
            sc_for_lbl = max(sc for l2, _d, sc in top3 if l2 == lbl)
            return int(lbl), float(sc_for_lbl), ranked

    return -1, 0.0, ranked   # gesture not recognized


def _dollar_predict_one(cand_pre: np.ndarray,
                         tmpl_pre: list,
                         tmpl_lbl: list,
                         k: int = KNN_K,
                         allow_rejection: bool = False,
                         epsilon: float = DOLLAR_EPSILON) -> int:
    """1-NN prediction via dollar_recognize (includes GSS + optional heuristic).
    Returns the predicted label, or -1 if allow_rejection=True and no gesture
    passes the heuristic thresholds.
    """
    label, _score, _ranked = dollar_recognize(
        cand_pre, tmpl_pre, tmpl_lbl,
        allow_rejection=allow_rejection, epsilon=epsilon)
    return label
