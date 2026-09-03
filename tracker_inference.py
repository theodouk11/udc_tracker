"""ByteTrack-style multi-object tracker on top of the fine-tuned GroundingDINO detector.

Two-stage per-frame association (see README_urinebag.md §13):
  1. Match existing tracks (Kalman-predicted position) against high-confidence
     detections (score >= score_thr) via IoU + Hungarian assignment.
  2. For tracks still unmatched, attempt a second match against low-confidence
     detections (low_thr <= score < score_thr) that would normally be discarded.
     A spatially-consistent low-confidence box is accepted ("recovered") because
     track continuity corroborates it.

Tracks are class-agnostic for association (the 4 phrases are different states of
the same physical object, so a flicker between labels must not break identity);
the displayed label per track is a temporal majority vote over its recent history.
Generalizes to N simultaneous objects, not hardcoded to a single bag.
"""

import argparse
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mmdet.apis import init_detector, inference_detector

# Real target classes we actually want to track.
TARGET_CLASSES = [
    "covered urine bag",
    "empty urine bag",
    "full urine bag",
    "half full urine bag",
]

# Distractor phrases: not tracked, just present in the prompt so the model has
# somewhere else to put a background lookalike (bottle, case, etc.) instead of
# misfiring one of the 4 real classes on it (§6). Detections with these labels
# are discarded before they ever reach the tracker's association step.
DISTRACTOR_CLASSES = [
    "plastic bottle",
    "water bottle",
    "folded shirt",
    "clock",
]

CLASS_NAMES = TARGET_CLASSES + DISTRACTOR_CLASSES

PALETTE = {
    "covered urine bag": (60, 20, 220),
    "empty urine bag": (75, 180, 60),
    "full urine bag": (0, 165, 255),
    "half full urine bag": (200, 130, 0),
}


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expand_box(box, margin):
    x1, y1, x2, y2 = box
    return np.array([x1 - margin, y1 - margin, x2 + margin, y2 + margin])


def boxes_near(a, b, margin):
    """True if a and b intersect once both are expanded by margin pixels.
    Independent sigmoid scoring means a confident distractor detection doesn't
    reduce a nearby target-class score on its own (verified empirically: a
    "water bottle" box can score 0.75 a few pixels from a "half full urine bag"
    box scoring 0.61, non-overlapping, both from the same physical object) —
    this proximity check is what actually implements the intended suppression."""
    ax1, ay1, ax2, ay2 = expand_box(a, margin)
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def crop_hist(frame, bbox_xyxy, bins=32):
    """HSV color histogram of a box's crop — a cheap appearance descriptor
    for re-identifying an object across a tracking gap (occlusion, or the
    camera cutting to a different angle) where motion/IoU continuity is
    unavailable. Reuses the same fill-color signal already established as
    the model's real detection cue (README §15-17)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def hist_similarity(a, b):
    if a is None or b is None:
        return -1.0
    return cv2.compareHist(a, b, cv2.HISTCMP_CORREL)


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over [cx, cy, w, h]."""

    def __init__(self, bbox_xyxy):
        # state: [cx, cy, w, h, vcx, vcy, vw, vh]
        self.x = np.zeros(8)
        self.x[:4] = xyxy_to_cxcywh(bbox_xyxy)

        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # constant-velocity coupling

        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # high initial uncertainty on velocity

        q = np.ones(8)
        q[:4] = 1.0     # position/size process noise
        q[4:] = 0.05     # velocity process noise (bag doesn't accelerate much)
        self.Q = np.diag(q)

        self.R = np.diag([2.0, 2.0, 4.0, 4.0])  # measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return cxcywh_to_xyxy(self.x[:4])

    def update(self, bbox_xyxy):
        z = xyxy_to_cxcywh(bbox_xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def current_bbox(self):
        return cxcywh_to_xyxy(self.x[:4])


class Track:
    _next_id = 1

    def __init__(self, bbox_xyxy, label, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanBoxTracker(bbox_xyxy)
        self.time_since_update = 0
        self.hit_streak = 0
        self.confirmed = False
        self.label_history = deque(maxlen=15)  # ~0.5s at 30fps
        self.label_history.append(label)
        self.last_score = score
        self.last_bbox = bbox_xyxy
        self.recovered_last = False
        self.match_history = deque(maxlen=HIT_RATE_WINDOW)  # True/False per frame: matched or not
        self.score_history = deque(maxlen=HIT_RATE_WINDOW)  # score on matched frames only
        self.peak_score = score  # highest score this track has EVER hit (lifetime, not windowed)
        self.appearance = None  # running HSV-histogram descriptor, set via update_appearance()
        self.n_updates = 0        # cumulative successful matches (survives gaps, unlike hit_streak)
        self.reid_checked = False  # whether this track has already had its one gallery-match attempt

    def predict(self):
        bbox = self.kf.predict()
        self.last_bbox = bbox
        return bbox

    def update(self, bbox_xyxy, label, score, recovered=False):
        self.kf.update(bbox_xyxy)
        self.last_bbox = self.kf.current_bbox()
        self.time_since_update = 0
        self.hit_streak += 1
        if self.hit_streak >= MIN_HITS:
            self.confirmed = True
        self.label_history.append(label)
        self.last_score = score
        self.recovered_last = recovered
        self.match_history.append(True)
        self.score_history.append(score)
        self.peak_score = max(self.peak_score, score)
        self.n_updates += 1

    def update_appearance(self, frame, alpha=0.3):
        """Blend this frame's crop into a running appearance descriptor
        (EMA, not just the latest frame) so a single blurry/oddly-lit frame
        can't dominate what gets compared at re-ID time."""
        hist = crop_hist(frame, self.last_bbox)
        if hist is None:
            return
        self.appearance = hist if self.appearance is None else (alpha * hist + (1 - alpha) * self.appearance)

    def mark_missed(self):
        self.time_since_update += 1
        self.hit_streak = 0
        self.recovered_last = False
        self.match_history.append(False)

    def smoothed_label(self):
        return Counter(self.label_history).most_common(1)[0][0]

    def hit_rate(self):
        if not self.match_history:
            return 0.0
        return sum(self.match_history) / len(self.match_history)

    def avg_score(self):
        if not self.score_history:
            return 0.0
        return sum(self.score_history) / len(self.score_history)

    def is_displayable(self):
        """Confirmed AND still being consistently matched AND has, at some
        point in its life, proven itself with a genuinely confident detection.
        hit-rate catches tracks sustained only by sparse, occasional matches
        with long gaps. peak_score is a lifetime (not rolling-window) high
        score: once a track clears PROOF_THR even once, it stays "proven" and
        rides out weak/blurry stretches on hit-rate alone afterward — this is
        the whole point of low-confidence recovery. A flat rolling-average
        threshold was tried first and rejected: it punishes a real object's
        temporarily weak patch almost as much as a false positive that is
        ALWAYS weak, because it forgets a strong detection once it ages out
        of the window. A persistent shortcut-driven false positive (a
        crumpled shirt scoring 0.1-0.44 on "empty urine bag" every frame)
        never spikes past PROOF_THR at all, so it never earns proof."""
        if not self.confirmed:
            return False
        return self.hit_rate() >= HIT_RATE_THR and self.peak_score >= PROOF_THR


MIN_HITS = 3
MAX_AGE = 30
IOU_THR_HIGH = 0.3
IOU_THR_LOW = 0.25
DUP_IOU_THR = 0.3  # don't spawn a new track that overlaps an existing predicted track this much
HIT_RATE_WINDOW = 30    # ~1s at 30fps rolling window
HIT_RATE_THR = 0.4      # must be matched (high or low-conf) in >=40% of the recent window
DISTRACTOR_VETO_THR = 0.15  # distractor score needed to veto a nearby target-class box
VETO_MARGIN = 30            # px each box is expanded by before checking proximity
PROOF_THR = 0.5              # lifetime peak score a track must hit once to be "proven" real
GALLERY_TTL = 1800           # ~60s at 30fps: how long a dead track's appearance is kept for re-ID
REID_CHECK_AFTER = 20        # frames a new track accumulates its own EMA before attempting re-ID
REID_SIMILARITY_THR = 0.25   # min HSV-histogram correlation to revive a dead track's identity


class ByteTracker:
    def __init__(self):
        self.tracks = []
        # recently-dead, previously-proven tracks kept around for re-ID by
        # appearance: {id, label_history, peak_score, appearance, age}
        self.lost_gallery = []
        self.last_revivals = []  # (old_id, similarity) pairs from the most recent step()

    def _associate(self, track_boxes, det_boxes, iou_thr):
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        cost = np.zeros((len(track_boxes), len(det_boxes)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                cost[i, j] = 1.0 - iou_xyxy(tb, db)

        row_ind, col_ind = linear_sum_assignment(cost)
        matches, um_tracks, um_dets = [], [], []
        matched_t, matched_d = set(), set()
        for r, c in zip(row_ind, col_ind):
            if 1.0 - cost[r, c] >= iou_thr:
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)
        um_tracks = [i for i in range(len(track_boxes)) if i not in matched_t]
        um_dets = [j for j in range(len(det_boxes)) if j not in matched_d]
        return matches, um_tracks, um_dets

    def step(self, frame, high_dets, low_dets):
        """high_dets/low_dets: list of (bbox_xyxy, label, score)."""
        predicted = [t.predict() for t in self.tracks]

        # stage 1: existing tracks vs high-confidence detections
        high_boxes = [d[0] for d in high_dets]
        matches1, um_tracks1, um_high = self._associate(predicted, high_boxes, IOU_THR_HIGH)
        for ti, di in matches1:
            bbox, label, score = high_dets[di]
            self.tracks[ti].update(bbox, label, score, recovered=False)
            self.tracks[ti].update_appearance(frame)

        # stage 2: still-unmatched tracks vs low-confidence detections (recovery)
        um_track_boxes = [predicted[i] for i in um_tracks1]
        low_boxes = [d[0] for d in low_dets]
        matches2, um_tracks2_rel, _ = self._associate(um_track_boxes, low_boxes, IOU_THR_LOW)
        recovered_track_idx = set()
        for ti_rel, di in matches2:
            ti = um_tracks1[ti_rel]
            bbox, label, score = low_dets[di]
            self.tracks[ti].update(bbox, label, score, recovered=True)
            self.tracks[ti].update_appearance(frame)
            recovered_track_idx.add(ti)

        # tracks unmatched in both stages
        still_unmatched = [um_tracks1[i] for i in um_tracks2_rel]
        for ti in still_unmatched:
            self.tracks[ti].mark_missed()

        # spawn new tracks from unmatched high-confidence detections that don't
        # overlap any current track's predicted position (avoid duplicate tracks)
        all_predicted = predicted  # predictions from this frame, pre-update
        for di in um_high:
            bbox, label, score = high_dets[di]
            overlaps_existing = any(iou_xyxy(bbox, pb) >= DUP_IOU_THR for pb in all_predicted)
            if not overlaps_existing:
                new_track = Track(bbox, label, score)
                new_track.update_appearance(frame)
                self.tracks.append(new_track)

        # re-ID pass: a candidate's FIRST-frame appearance is far too noisy to
        # compare reliably (verified empirically: real same-object match scored
        # -0.003 on a single raw frame, indistinguishable from noise). Waiting
        # until its own running EMA has ~20 frames to converge makes the same
        # comparison highly reliable (0.45 for a genuine match vs <0.09 for an
        # unrelated object at that point) — so each new track gets exactly ONE
        # gallery-match attempt, deferred until it has enough of its own
        # accumulated appearance to compare against.
        self.last_revivals = []
        for t in self.tracks:
            if t.reid_checked or t.n_updates < REID_CHECK_AFTER or t.appearance is None:
                continue
            t.reid_checked = True
            best_entry, best_sim = None, -1.0
            for entry in self.lost_gallery:
                sim = hist_similarity(t.appearance, entry["appearance"])
                if sim > best_sim:
                    best_entry, best_sim = entry, sim
            if best_entry is not None and best_sim >= REID_SIMILARITY_THR:
                # revive: reuse the old identity, but keep this track's own
                # (already well-established, 20-frames-proven) motion state —
                # only identity/proof carries over, not stale position
                old_id = t.id
                t.id = best_entry["id"]
                t.peak_score = max(t.peak_score, best_entry["peak_score"])
                self.lost_gallery.remove(best_entry)
                self.last_revivals.append((best_entry["id"], old_id, best_sim))

        # tracks that just crossed MAX_AGE this step: if they were ever proven
        # real, keep their identity/appearance around for possible re-ID
        just_died = [t for t in self.tracks if t.time_since_update > MAX_AGE]
        for t in just_died:
            if t.peak_score >= PROOF_THR and t.appearance is not None:
                self.lost_gallery.append({
                    "id": t.id,
                    "label_history": deque(t.label_history, maxlen=15),
                    "peak_score": t.peak_score,
                    "appearance": t.appearance,
                    "age": 0,
                })

        # age out gallery entries and prune dead tracks
        for entry in self.lost_gallery:
            entry["age"] += 1
        self.lost_gallery = [e for e in self.lost_gallery if e["age"] <= GALLERY_TTL]
        self.tracks = [t for t in self.tracks if t.time_since_update <= MAX_AGE]

        return self.tracks


def run_tracker_on_video(model, in_path, out_path, score_thr, low_thr, debug_log=None):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    tracker = ByteTracker()
    frame_idx = 0
    class_hits = {c: 0 for c in TARGET_CLASSES}
    recovered_frames = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = inference_detector(model, frame, text_prompt=CLASS_NAMES, custom_entities=True)
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()

        # distractor detections are never tracked themselves; they're only used
        # below to veto nearby target-class candidates that are likely the same
        # physical (non-bag) object
        distractor_mask = (labels >= len(TARGET_CLASSES)) & (scores >= DISTRACTOR_VETO_THR)
        distractor_boxes = bboxes[distractor_mask]

        target_mask = labels < len(TARGET_CLASSES)
        high_mask = target_mask & (scores >= score_thr)
        low_mask = target_mask & (scores >= low_thr) & (scores < score_thr)

        def not_vetoed(i):
            return not any(boxes_near(db, bboxes[i], VETO_MARGIN) for db in distractor_boxes)

        high_dets = [(bboxes[i], CLASS_NAMES[labels[i]], scores[i])
                     for i in np.where(high_mask)[0] if not_vetoed(i)]
        low_dets = [(bboxes[i], CLASS_NAMES[labels[i]], scores[i])
                    for i in np.where(low_mask)[0] if not_vetoed(i)]

        tracks = tracker.step(frame, high_dets, low_dets)
        for revived_id, temp_id, sim in tracker.last_revivals:
            print(f"  [{os.path.basename(in_path)}] frame {frame_idx}: track #{temp_id} "
                  f"revived as #{revived_id} (appearance similarity {sim:.3f})", flush=True)

        # unconditional debug logging: every track updated this frame, regardless
        # of whether it currently passes the display gate, so gate logic can be
        # tuned/analyzed offline from one recorded run instead of rerunning the model
        if debug_log is not None:
            for t in tracks:
                if t.time_since_update != 0:
                    continue  # only log frames where this track actually got a detection
                x1, y1, x2, y2 = t.last_bbox.astype(int)
                debug_log.write(f"{os.path.basename(in_path)},{frame_idx},{t.id},{t.smoothed_label()},"
                                 f"{t.last_score:.3f},{t.recovered_last},{t.is_displayable()},"
                                 f"{t.hit_rate():.3f},{t.avg_score():.3f},{t.peak_score:.3f},"
                                 f"{x1},{y1},{x2},{y2}\n")

        frame_recovered = False
        for t in tracks:
            if not t.is_displayable():
                continue
            label = t.smoothed_label()
            color = PALETTE[label]
            x1, y1, x2, y2 = t.last_bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            tag = " [rec]" if t.recovered_last else ""
            text = f"#{t.id} {label}: {t.last_score:.2f}{tag}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            class_hits[label] += 1
            if t.recovered_last:
                frame_recovered = True

        if frame_recovered:
            recovered_frames += 1

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == n_frames:
            elapsed = time.time() - t0
            rate = frame_idx / elapsed
            eta = (n_frames - frame_idx) / rate if rate > 0 else 0
            print(f"  [{os.path.basename(in_path)}] frame {frame_idx}/{n_frames} "
                  f"({rate:.1f} fps, ETA {eta:.0f}s)", flush=True)

    cap.release()
    writer.release()
    return frame_idx, class_hits, recovered_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_thr", type=float, default=0.3,
                         help="stage-1 (high-confidence) acceptance threshold")
    parser.add_argument("--low_thr", type=float, default=0.05,
                         help="stage-2 (low-confidence recovery) floor")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debug_log", default=None,
                         help="if set, write a CSV of every displayed box (video,frame,id,label,score,recovered,x1,y1,x2,y2)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)

    videos = sorted(f for f in os.listdir(args.video_dir) if f.lower().endswith(".mp4"))
    print(f"Found {len(videos)} videos: {videos}", flush=True)

    debug_log = open(args.debug_log, "w") if args.debug_log else None
    if debug_log is not None:
        debug_log.write("video,frame,track_id,label,score,recovered,displayable,hit_rate,avg_score,peak_score,x1,y1,x2,y2\n")

    summary = []
    for v in videos:
        Track._next_id = 1  # reset IDs per video
        in_path = os.path.join(args.video_dir, v)
        out_path = os.path.join(args.out_dir, f"tracked_{v}")
        print(f"Processing {v} -> {out_path}", flush=True)
        n_frames, class_hits, recovered_frames = run_tracker_on_video(
            model, in_path, out_path, args.score_thr, args.low_thr, debug_log=debug_log)
        summary.append((v, n_frames, class_hits, recovered_frames))

    if debug_log is not None:
        debug_log.close()

    print("\n=== Summary ===")
    for v, n_frames, class_hits, recovered_frames in summary:
        total_hits = sum(class_hits.values())
        print(f"{v}: {n_frames} frames, {total_hits} confirmed-track detections, "
              f"{recovered_frames} frames with a low-confidence recovery "
              f"({100 * recovered_frames / n_frames:.1f}%)")
        for c, cnt in class_hits.items():
            if cnt > 0:
                print(f"    {c}: detected in {cnt}/{n_frames} frames "
                      f"({100 * cnt / n_frames:.1f}%)")


if __name__ == "__main__":
    main()
