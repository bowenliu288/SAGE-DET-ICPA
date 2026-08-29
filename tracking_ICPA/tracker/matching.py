import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import math
# ============================================================
# SparseTrack Windows fallback for cython_bbox
#
# The original code imports:
#     from cython_bbox import bbox_overlaps as bbox_ious
#
# On Windows, cython_bbox often requires a local C/C++ build.
# To keep the SparseTrack association logic unchanged, use the
# original implementation when available, otherwise fall back
# to a NumPy implementation with the same bbox-overlap API.
# ============================================================
try:
    from cython_bbox import bbox_overlaps as bbox_ious
    CYTHON_BBOX_AVAILABLE = True
except ImportError:
    CYTHON_BBOX_AVAILABLE = False

    def bbox_ious(boxes, query_boxes):
        """
        NumPy fallback for cython_bbox.bbox_overlaps.

        Parameters
        ----------
        boxes : ndarray, shape [N, 4]
        query_boxes : ndarray, shape [K, 4]

        Returns
        -------
        overlaps : ndarray, shape [N, K]

        Notes
        -----
        This follows the standard cython_bbox coordinate
        convention using inclusive box extents (+1).
        """
        boxes = np.asarray(
            boxes,
            dtype=np.float64
        )

        query_boxes = np.asarray(
            query_boxes,
            dtype=np.float64
        )

        n = boxes.shape[0]
        k = query_boxes.shape[0]

        overlaps = np.zeros(
            (n, k),
            dtype=np.float64
        )

        if n == 0 or k == 0:
            return overlaps

        box_w = np.maximum(
            0.0,
            boxes[:, 2] - boxes[:, 0] + 1.0
        )

        box_h = np.maximum(
            0.0,
            boxes[:, 3] - boxes[:, 1] + 1.0
        )

        box_area = box_w * box_h

        query_w = np.maximum(
            0.0,
            query_boxes[:, 2]
            - query_boxes[:, 0]
            + 1.0
        )

        query_h = np.maximum(
            0.0,
            query_boxes[:, 3]
            - query_boxes[:, 1]
            + 1.0
        )

        query_area = query_w * query_h

        xx1 = np.maximum(
            boxes[:, None, 0],
            query_boxes[None, :, 0]
        )

        yy1 = np.maximum(
            boxes[:, None, 1],
            query_boxes[None, :, 1]
        )

        xx2 = np.minimum(
            boxes[:, None, 2],
            query_boxes[None, :, 2]
        )

        yy2 = np.minimum(
            boxes[:, None, 3],
            query_boxes[None, :, 3]
        )

        iw = np.maximum(
            0.0,
            xx2 - xx1 + 1.0
        )

        ih = np.maximum(
            0.0,
            yy2 - yy1 + 1.0
        )

        inter = iw * ih

        union = (
            box_area[:, None]
            + query_area[None, :]
            - inter
        )

        valid = union > 0

        overlaps[valid] = (
            inter[valid]
            / union[valid]
        )

        return overlaps

from tracker import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost



# ============================================================
# ICPA-SparseTrack v1
# IoU-Center Pixel-Aware Association
#
# The SparseTrack native DCM structure is NOT changed.
# This function is called only inside the high-score DCM
# depth subset.
#
# Native branch:
#     SparseTrack own IoU distance
#     + original score fusion when enabled
#
# Complementary branch:
#     locally normalized center-distance
#
# Match orientation:
#     [track_index, detection_index]
# ============================================================

def _icpa_sparse_tlbr_array(items):
    """
    Convert SparseTrack track/detection objects or ndarray
    into an [N, 4] tlbr array.
    """
    if len(items) == 0:
        return np.zeros(
            (0, 4),
            dtype=np.float32
        )

    if isinstance(items, np.ndarray):
        arr = np.asarray(
            items,
            dtype=np.float32
        )
        return arr[:, :4]

    if hasattr(items[0], "tlbr"):
        return np.asarray(
            [
                item.tlbr
                for item in items
            ],
            dtype=np.float32
        )

    arr = np.asarray(
        items,
        dtype=np.float32
    )

    return arr[:, :4]


def _icpa_sparse_box_area(tlbr):
    """
    Area of [x1, y1, x2, y2].
    """
    w = max(
        0.0,
        float(
            tlbr[2]
            - tlbr[0]
        )
    )

    h = max(
        0.0,
        float(
            tlbr[3]
            - tlbr[1]
        )
    )

    return w * h


def center_distance_matrix_icpa_sparse(
        tracks,
        detections
):
    """
    Locally normalized center-distance cost.

    d_center /
    sqrt(
        (w_track + w_det)^2
        +
        (h_track + h_det)^2
    )

    Smaller is better.
    """
    track_boxes = _icpa_sparse_tlbr_array(
        tracks
    )

    det_boxes = _icpa_sparse_tlbr_array(
        detections
    )

    num_tracks = len(
        track_boxes
    )

    num_dets = len(
        det_boxes
    )

    if (
        num_tracks == 0
        or
        num_dets == 0
    ):
        return np.zeros(
            (
                num_tracks,
                num_dets
            ),
            dtype=np.float32
        )

    trk_cx = (
        track_boxes[:, 0]
        + track_boxes[:, 2]
    ) / 2.0

    trk_cy = (
        track_boxes[:, 1]
        + track_boxes[:, 3]
    ) / 2.0

    trk_w = np.maximum(
        0.0,
        track_boxes[:, 2]
        - track_boxes[:, 0]
    )

    trk_h = np.maximum(
        0.0,
        track_boxes[:, 3]
        - track_boxes[:, 1]
    )

    det_cx = (
        det_boxes[:, 0]
        + det_boxes[:, 2]
    ) / 2.0

    det_cy = (
        det_boxes[:, 1]
        + det_boxes[:, 3]
    ) / 2.0

    det_w = np.maximum(
        0.0,
        det_boxes[:, 2]
        - det_boxes[:, 0]
    )

    det_h = np.maximum(
        0.0,
        det_boxes[:, 3]
        - det_boxes[:, 1]
    )

    dx = (
        trk_cx[:, None]
        - det_cx[None, :]
    )

    dy = (
        trk_cy[:, None]
        - det_cy[None, :]
    )

    center_dist = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    local_scale = np.sqrt(
        (
            trk_w[:, None]
            + det_w[None, :]
        ) ** 2
        +
        (
            trk_h[:, None]
            + det_h[None, :]
        ) ** 2
    )

    local_scale = np.maximum(
        local_scale,
        1e-6
    )

    cost = (
        center_dist
        / local_scale
    )

    cost = np.clip(
        cost,
        0.0,
        1.0
    )

    return cost.astype(
        np.float32
    )


def parallel_iou_center_assignment_icpa_sparse(
        tracks,
        detections,
        match_thresh=0.75,
        center_thresh=0.70,
        small_area=32 * 32,
        motion_thresh=0.70,
        fuse_score_flag=True
):
    """
    ICPA for one SparseTrack DCM depth subset.

    Two independent assignments:
      1) SparseTrack native IoU association
         (+ original score fusion when enabled)
      2) normalized center-distance association

    Consensus pairs are accepted first. For branch conflicts,
    center geometry is preferred when:
      - the detection is a small target, or
      - raw IoU overlap is strongly degraded.

    NOTE:
    ``motion_thresh`` is retained only for compatibility with
    the existing ICPA implementation. It is applied to
    IoU distance (1 - IoU), not physical motion.
    """
    num_tracks = len(
        tracks
    )

    num_dets = len(
        detections
    )

    if (
        num_tracks == 0
        or
        num_dets == 0
    ):
        matches = np.empty(
            (0, 2),
            dtype=np.int64
        )

        u_track = np.arange(
            num_tracks,
            dtype=np.int64
        )

        u_det = np.arange(
            num_dets,
            dtype=np.int64
        )

        return (
            matches,
            u_track,
            u_det
        )

    # --------------------------------------------------------
    # SparseTrack native IoU branch.
    # Use SparseTrack's own iou_distance().
    # --------------------------------------------------------
    raw_iou_cost = iou_distance(
        tracks,
        detections
    )

    native_cost = (
        raw_iou_cost.copy()
    )

    if fuse_score_flag:
        native_cost = fuse_score(
            native_cost,
            detections
        )

    matches_native, _, _ = (
        linear_assignment(
            native_cost,
            match_thresh
        )
    )

    # --------------------------------------------------------
    # Independent center branch.
    # --------------------------------------------------------
    center_cost = (
        center_distance_matrix_icpa_sparse(
            tracks,
            detections
        )
    )

    matches_center, _, _ = (
        linear_assignment(
            center_cost,
            center_thresh
        )
    )

    matches_native = np.asarray(
        matches_native,
        dtype=np.int64
    ).reshape(-1, 2)

    matches_center = np.asarray(
        matches_center,
        dtype=np.int64
    ).reshape(-1, 2)

    set_native = set(
        (
            int(i),
            int(j)
        )
        for i, j
        in matches_native
    )

    set_center = set(
        (
            int(i),
            int(j)
        )
        for i, j
        in matches_center
    )

    same_matches_set = (
        set_native
        & set_center
    )

    union_matches_set = (
        set_native
        | set_center
    )

    final_matches = []

    used_tracks = set()
    used_dets = set()

    # --------------------------------------------------------
    # 1. Consensus pairs first.
    # --------------------------------------------------------
    same_matches = sorted(
        list(
            same_matches_set
        ),
        key=lambda pair:
            float(
                native_cost[
                    pair[0],
                    pair[1]
                ]
            )
    )

    for trk_i, det_i in same_matches:

        if (
            trk_i not in used_tracks
            and
            det_i not in used_dets
        ):
            final_matches.append(
                [
                    trk_i,
                    det_i
                ]
            )

            used_tracks.add(
                trk_i
            )

            used_dets.add(
                det_i
            )

    # --------------------------------------------------------
    # 2. Resolve branch conflicts.
    # --------------------------------------------------------
    candidates = []

    det_tlbrs = (
        _icpa_sparse_tlbr_array(
            detections
        )
    )

    safety_iou_cost_gate = min(
        float(match_thresh)
        + 0.15,
        0.98
    )

    for trk_i, det_i in union_matches_set:

        if (
            trk_i,
            det_i
        ) in same_matches_set:
            continue

        if (
            trk_i in used_tracks
            or
            det_i in used_dets
        ):
            continue

        det_area = (
            _icpa_sparse_box_area(
                det_tlbrs[
                    det_i
                ]
            )
        )

        overlap_degradation = float(
            raw_iou_cost[
                trk_i,
                det_i
            ]
        )

        prefer_center = (
            det_area
            < float(small_area)
            or
            overlap_degradation
            > float(motion_thresh)
        )

        from_center = (
            trk_i,
            det_i
        ) in set_center

        from_native = (
            trk_i,
            det_i
        ) in set_native

        if (
            prefer_center
            and
            from_center
        ):
            priority = 0

            cost_value = float(
                center_cost[
                    trk_i,
                    det_i
                ]
            )

        elif (
            (not prefer_center)
            and
            from_native
        ):
            priority = 0

            cost_value = float(
                native_cost[
                    trk_i,
                    det_i
                ]
            )

        elif from_center:
            priority = 1

            cost_value = float(
                center_cost[
                    trk_i,
                    det_i
                ]
            )

        else:
            priority = 1

            cost_value = float(
                native_cost[
                    trk_i,
                    det_i
                ]
            )

        # Conservative safety gate.
        # Reject only when both overlap and center geometry
        # are unreliable.
        if (
            raw_iou_cost[
                trk_i,
                det_i
            ]
            > safety_iou_cost_gate
            and
            center_cost[
                trk_i,
                det_i
            ]
            > float(center_thresh)
        ):
            continue

        candidates.append(
            (
                priority,
                cost_value,
                trk_i,
                det_i
            )
        )

    candidates.sort(
        key=lambda item:
            (
                item[0],
                item[1]
            )
    )

    for (
        _,
        _,
        trk_i,
        det_i
    ) in candidates:

        if (
            trk_i not in used_tracks
            and
            det_i not in used_dets
        ):
            final_matches.append(
                [
                    trk_i,
                    det_i
                ]
            )

            used_tracks.add(
                trk_i
            )

            used_dets.add(
                det_i
            )

    matches = np.asarray(
        final_matches,
        dtype=np.int64
    ).reshape(-1, 2)

    u_track = np.asarray(
        [
            i
            for i
            in range(num_tracks)
            if i
            not in used_tracks
        ],
        dtype=np.int64
    )

    u_det = np.asarray(
        [
            j
            for j
            in range(num_dets)
            if j
            not in used_dets
        ],
        dtype=np.int64
    )

    return (
        matches,
        u_track,
        u_det
    )

def greedy_assignment_iou(dist, thresh):
        matched_indices = []
        if dist.shape[1] == 0:
            return np.array(matched_indices, np.int32).reshape(-1, 2)
        for i in range(dist.shape[0]):
            j = dist[i].argmin()
            if dist[i][j] < thresh:
                dist[:, j] = 1.
                matched_indices.append([j, i])
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    
def greedy_assignment(dists, threshs):
    matches = greedy_assignment_iou(dists.T, threshs)
    u_det = [d for d in range(dists.shape[1]) if not (d in matches[:, 1])]
    u_track = [d for d in range(dists.shape[0]) if not (d in matches[:, 0])]
    return matches, u_track,  u_det

def fuse_score_matrix(cost_matrix, detections, tracks):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    trk_scores = np.array([trk.score for trk in tracks])
    trk_scores = np.expand_dims(trk_scores, axis=1).repeat(cost_matrix.shape[1], axis=1)
    mid_scores = (det_scores + trk_scores) / 2
    fuse_sim = iou_sim * mid_scores
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost

def BIoU_distance(atracks, btracks, sigma = 0.4):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    atlbrs, btlbrs = [], []
    for trk in atracks:
        x1,y1,w,h = trk.tlwh
        delta_h, delta_w = h * sigma, w * sigma
        x1_ = x1 - delta_w
        y1_ = y1 - delta_h
        x2_ = x1 + w + delta_w
        y2_ = y1 + h + delta_h
        bbox_new = np.array([x1_, y1_, x2_, y2_], dtype=np.float32)
        atlbrs.append(bbox_new)
        
    for trk in btracks:
        x1,y1,w,h = trk.tlwh
        delta_h, delta_w = h * sigma, w * sigma
        x1_ = x1 - delta_w
        y1_ = y1 - delta_h
        x2_ = x1 + w + delta_w
        y2_ = y1 + h + delta_h
        bbox_new = np.array([x1_, y1_, x2_, y2_], dtype=np.float32)
        btlbrs.append(bbox_new)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix