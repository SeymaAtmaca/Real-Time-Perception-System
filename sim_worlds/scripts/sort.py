import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    if np.any(np.isnan(bb_test)) or np.any(np.isnan(bb_gt)):
        return 0.0

    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
        (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh
    )
    return o


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4,1))


def convert_x_to_bbox(x, score=None):
    s = max(x[2, 0], 1e-6)
    r = max(x[3, 0], 1e-6)

    w = np.sqrt(s * r)
    h = s / w

    x1 = x[0, 0] - w / 2.
    y1 = x[1, 0] - h / 2.
    x2 = x[0, 0] + w / 2.
    y2 = x[1, 0] + h / 2.

    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))




class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = self._init_kf()
        self.kf['x'][:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def _init_kf(self):
        kf = {}
        kf['x'] = np.zeros((7,1))
        kf['P'] = np.eye(7) * 10.
        kf['F'] = np.eye(7)
        for i in range(4):
            kf['F'][i, i+3] = 1.
        kf['Q'] = np.eye(7)
        kf['R'] = np.eye(4)
        kf['H'] = np.zeros((4,7))
        kf['H'][:4, :4] = np.eye(4)
        return kf

    def predict(self):
        self.kf['x'] = self.kf['F'] @ self.kf['x']
        self.kf['P'] = self.kf['F'] @ self.kf['P'] @ self.kf['F'].T + self.kf['Q']
        self.age += 1
        self.time_since_update += 1

        bbox = convert_x_to_bbox(self.kf['x'])[0]
        if np.any(np.isnan(bbox)):
            return np.array([[0, 0, 0, 0]])

        return bbox.reshape(1, 4)


    def update(self, bbox):
        z = convert_bbox_to_z(bbox)
        y = z - self.kf['H'] @ self.kf['x']
        S = self.kf['H'] @ self.kf['P'] @ self.kf['H'].T + self.kf['R']
        K = self.kf['P'] @ self.kf['H'].T @ np.linalg.inv(S)
        self.kf['x'] += K @ y
        self.kf['P'] = self.kf['P'] - K @ self.kf['H'] @ self.kf['P']
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):
        return convert_x_to_bbox(self.kf['x'])


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):

    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0), dtype=int)
        )

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) > 0:
        matches = np.concatenate(matches, axis=0)
    else:
        matches = np.empty((0, 2), dtype=int)

    return (
        matches,
        np.array(unmatched_detections),
        np.array(unmatched_trackers)
    )



class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5))

        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 4))
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = pos

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i,:4]))

        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((trk.get_state()[0], [trk.id])).reshape(1, -1))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))
