import numpy as np
from scipy.optimize import linear_sum_assignment
from src.tracking.track import Track
from src.tracking.kalman import KalmanBoxTracker


def bbox_ious(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    ix1 = np.maximum(b1_x1[:, None], b2_x1)
    iy1 = np.maximum(b1_y1[:, None], b2_y1)
    ix2 = np.minimum(b1_x2[:, None], b2_x2)
    iy2 = np.minimum(b1_y2[:, None], b2_y2)
    
    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih
    
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1[:, None] + area2 - inter + 1e-6
    
    return inter / union

def iou_distance(boxes1, boxes2):
    return 1.0 - bbox_ious(boxes1, boxes2)

def cosine_distance(emb1, emb2):
    if len(emb1) == 0 or len(emb2) == 0:
        return np.zeros((len(emb1), len(emb2)))
    e1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-6)
    e2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-6)
    sim = np.dot(e1, e2.T)
    return np.clip(1.0 - sim, 0.0, 2.0)

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    # Use scipy to handle assignment robustly to bypass lapjv bugs
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            unmatched_a.remove(r)
            unmatched_b.remove(c)
            
    if len(matches) == 0:
        return np.empty((0, 2), dtype=int), unmatched_a, unmatched_b
        
    return np.array(matches), unmatched_a, unmatched_b

class MAVERICTracker:
    def __init__(self, iou_weight=0.5, reid_weight=0.5, max_age=30):
        self.tracks = []
        self.iou_weight = iou_weight
        self.reid_weight = reid_weight
        self.max_age = max_age
        self.track_id_counter = 1

    def update(self, bboxes, scores, embeddings=None):
        if len(bboxes) == 0:
            bboxes = np.empty((0, 4))
            scores = np.empty((0,))
        if embeddings is None:
            embeddings = np.zeros((len(bboxes), 512)) if len(bboxes) > 0 else np.empty((0, 512))

        # Step 1: Predict on all active tracks
        for t in self.tracks:
            t.predict()

        # Step 2: Split detections
        high_mask = scores > 0.6
        low_mask = (scores > 0.1) & (scores <= 0.6)
        
        high_bboxes = bboxes[high_mask]
        high_scores = scores[high_mask]
        high_embeddings = embeddings[high_mask]

        low_bboxes = bboxes[low_mask]
        low_scores = scores[low_mask]
        low_embeddings = embeddings[low_mask]

        # First pass uses all active tracks
        first_pass_tracks = self.tracks
        
        # Step 3: First pass (high-conf -> confirmed/active tracks)
        matches_first, unmatched_tracks_idx, unmatched_high_idx = [], list(range(len(first_pass_tracks))), list(range(len(high_bboxes)))
        
        if len(first_pass_tracks) > 0 and len(high_bboxes) > 0:
            track_boxes = np.array([t.kalman.get_state()[:4] for t in first_pass_tracks])
            track_embs = np.array([t.embedding for t in first_pass_tracks])
            
            iou_dists = iou_distance(track_boxes, high_bboxes)
            reid_dists = cosine_distance(track_embs, high_embeddings)
            
            cost_matrix = self.iou_weight * iou_dists + self.reid_weight * reid_dists
            matches_first, unmatched_tracks_idx, unmatched_high_idx = linear_assignment(cost_matrix, thresh=0.7)
            
            for trk_idx, det_idx in matches_first:
                first_pass_tracks[trk_idx].update(high_bboxes[det_idx], high_embeddings[det_idx])

        unmatched_tracks = []
        for i in unmatched_tracks_idx:
            track = first_pass_tracks[i]
            unmatched_tracks.append(track)
        
        # Step 4: Second pass (low-conf -> remaining unmatched confirmed tracks)
        second_pass_tracks_conf = [t for t in unmatched_tracks if t.state == "confirmed"]
        
        if len(second_pass_tracks_conf) > 0 and len(low_bboxes) > 0:
            track_boxes = np.array([t.kalman.get_state()[:4] for t in second_pass_tracks_conf])
            cost_matrix_low = iou_distance(track_boxes, low_bboxes)
            matches_second, _, _ = linear_assignment(cost_matrix_low, thresh=0.5)
            
            for trk_idx, det_idx in matches_second:
                second_pass_tracks_conf[trk_idx].update(low_bboxes[det_idx], low_embeddings[det_idx])

        # Step 5: Handle unmatched (unmatched high_conf -> new tentative tracks)
        for idx in unmatched_high_idx:
            kalman = KalmanBoxTracker(high_bboxes[idx])
            new_track = Track(id=self.track_id_counter, kalman=kalman, embedding=high_embeddings[idx], state="tentative", hits=1, age=1, time_since_update=0)
            self.track_id_counter += 1
            self.tracks.append(new_track)
            
        # Step 5 (cont): Handle unmatched tracks: keep alive if time_since_update <= max_age
        self.tracks = [t for t in self.tracks if not t.is_dead(self.max_age)]
        
        # Step 6: Return all confirmed tracks with bboxes and IDs
        # Only return tracks that were recently updated to avoid dumping stale predictions as false positives
        return [t for t in self.tracks if t.state == "confirmed" and t.time_since_update <= 1]
