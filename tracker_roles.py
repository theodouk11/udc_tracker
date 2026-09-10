"""
Joint tracker for the 7-class model: runs the urine-bag fill-state
pipeline and the staff role-tracking pipeline off a single
inference_detector call per frame, drawing both onto one output video.

  - Urine-bag pipeline (reused as-is from tracker_inference.py): peak-score
    proof gate, distractor veto, re-ID gallery, plus two additive gates -
    bed anchoring (a bag only displays while linked to a currently-tracked
    bed, killing false positives with no bed in view) and state hysteresis
    (a label must dominate a multi-second window before it's allowed to
    change, stopping noisy flips).

  - Role pipeline (person / badge / white coat, fully independent track
    pools): badge/coat bound to a person by containment (not proximity)
    via Hungarian matching, bonds persist through brief occlusion via
    assign_sticky() and auto-release if unsupported for ~10 frames, role
    evidence accumulates over a rolling window and locks once stable, and
    a one-way geometric veto (coat_extent_ratio) removes shirt/polo
    misdetections of "white coat" without ever inventing a coat that
    wasn't there.

The two pipelines share no mutable state and only interact by being drawn
onto the same frame. Threshold rationale and tuning history live in the
project's private design notes, not in this file.
"""

import argparse
import contextlib
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mmdet.apis import init_detector, inference_detector
from tracker_inference import (
    TARGET_CLASSES as URINE_TARGET_CLASSES,
    DISTRACTOR_CLASSES as URINE_DISTRACTOR_CLASSES,
    PALETTE as URINE_PALETTE,
    DISTRACTOR_VETO_THR,
    VETO_MARGIN,
    boxes_near,
    iou_xyxy,
    expand_box,
    ByteTracker,
    Track,
)
from src_path import JOINT_URINEBAG_PATH  # noqa: F401  (keeps src_path importable/consistent)

STAFF_CLASSES = ["person", "badge", "white coat"]
BED_CLASSES = ["bed"]   # untrained open-vocab anchor class, used only as a gate (v4) - never displayed

# Single combined prompt: urine-bag targets + urine-bag's own distractor
# phrases (unchanged, still only used to veto urine-bag candidates) + the
# 3 staff classes + the bed anchor class. Index ranges below are what keep
# the four groups apart.
CLASS_NAMES = URINE_TARGET_CLASSES + URINE_DISTRACTOR_CLASSES + STAFF_CLASSES + BED_CLASSES
N_URINE_TARGET = len(URINE_TARGET_CLASSES)
N_URINE_DISTRACTOR = len(URINE_DISTRACTOR_CLASSES)
N_STAFF = len(STAFF_CLASSES)
URINE_TARGET_END = N_URINE_TARGET
URINE_DISTRACTOR_END = N_URINE_TARGET + N_URINE_DISTRACTOR
STAFF_END = URINE_DISTRACTOR_END + N_STAFF

CONTAINMENT_THR = 0.5        # fraction of a badge/coat box that must fall inside a person box to count as worn
ROLE_EVIDENCE_WINDOW = 90    # ~3s at 30fps rolling window for role evidence (longer than the base tracker's
                              # HIT_RATE_WINDOW=30, since role should be sticky across longer detection gaps)
COAT_ROLE_THR = 0.3          # fraction of the recent window with an assigned coat needed to call "Doctor"
BADGE_ROLE_THR = 0.3         # fraction of the recent window with an assigned badge needed to call "Caregiver"
NMS_IOU_THR = 0.5            # same-class duplicate suppression: standard IoU overlap
NMS_CONTAINMENT_THR = 0.8    # same-class duplicate suppression: one box mostly nested inside the other
                              # (plain IoU-NMS misses this - a small box fully inside a big one has low IoU)
NMS_CONTAINMENT_DIST_RATIO = 0.6  # containment-only suppression additionally requires the two boxes'
                              # centers to be within this fraction of the SMALLER box's own diagonal.
                              # Found empirically (NMS/crowd probe): genuine same-object duplicate
                              # detections sit ~18px apart; false crowd merges (a big sloppy box in a
                              # dense scene swallowing a real, separate detection) sit 80-460px apart
                              # with containment still hitting 1.0 - IoU alone doesn't fire there (median
                              # 0.2-0.4), so containment needs its own sanity gate. IoU-triggered
                              # suppression is left ungated since the data showed it wasn't the culprit.
ROLE_CONFIRM_FRAMES = 30     # consecutive frames the SAME resolved role must hold before it's locked
                              # permanently - a person's real-world role never changes mid-video, so once
                              # genuinely sure, stop re-evaluating for the rest of the video
BOND_RELEASE_GRACE = 10      # consecutive frames a bonded pair's CURRENT containment must stay below
                              # CONTAINMENT_THR before assign_sticky releases the bond. Long enough to
                              # survive a brief 1-2 frame occlusion blip (the whole reason bonds are
                              # sticky), short enough to correct a bond that's become geometrically
                              # indefensible (observed directly: containment=0.0 for 40+ consecutive
                              # frames against the bonded person while a different, perfectly-contained
                              # candidate sat at 1.0 the entire time) instead of riding it out forever.
COAT_EXTENT_THR = 0.70       # (coat_bbox_bottom - person_bbox_top) / person_bbox_height below this looks
                              # shirt-like this frame (measured gap: real coats 0.741-0.815, shirt/polo
                              # misdetections 0.439-0.682 - see the coat-distractor probe)
MIN_PERSON_H_FOR_EXTENT = 60 # below this the matched person box is often a bad/spurious match
                              # (measure_coat_extent.py saw height<40px boxes give nonsense ratios)
EXTENT_VETO_RATE_THR = 0.5   # rolling rate (over ROLE_EVIDENCE_WINDOW) of "shirt-like" frames above
                              # which coat evidence is vetoed for that frame

BED_PROOF_THR = 0.25         # lifetime peak score a "bed" track must hit once to gate on - deliberately
                              # lower than the fine-tuned classes' PROOF_THR=0.5, since "bed" is an
                              # untrained open-vocab phrase (probe data: peak scores mostly 0.10-0.57,
                              # rarely above 0.5 even on a clearly-visible real bed)
BED_HIT_RATE_THR = 0.3       # rolling match rate a "bed" track needs to count as currently tracked
BED_ATTACH_THR = 0.5         # fraction of a bag box that must fall inside the (margin-expanded) bed
                              # box to count as "attached to this bed"
BED_ATTACH_MARGIN = 150      # px the bed box is expanded by before the attach containment check - the
                              # bag hangs from a clip just outside the bed's own tight detection box
                              # (confirmed directly on HALF-3/covered crops), not inside it like a badge
                              # sits inside a person box

STATE_HOLD_WINDOW = 150      # ~5s at 30fps: once a urine-bag track's displayed label is locked in, a
                              # DIFFERENT label must dominate this whole window before it's allowed to
                              # change - deliberately much longer than the base tracker's own 15-frame
                              # smoothed_label() window (that one absorbs ordinary per-frame flicker;
                              # this one exists specifically to resist a sustained-but-wrong stretch -
                              # confirmed directly on HALF-3.mp4: a ~100-130 frame wrong-label run at
                              # HIGHER confidence (0.86-0.92) than the correct label ever reached)
STATE_SWITCH_RATE_THR = 0.85 # once STATE_HOLD_WINDOW is full, the candidate label must be this
                              # dominant (not just plurality) to actually override the locked label

ROLE_PALETTE = {
    "Doctor": (255, 0, 255),      # BGR magenta
    "Caregiver": (0, 200, 0),     # BGR green
    "Person": (180, 180, 180),    # BGR grey - no confirmed role yet
    "badge": (0, 128, 128),
    "white coat": (128, 128, 0),
}


def containment(inner_xyxy, outer_xyxy):
    """Fraction of inner's own area that falls inside outer. Asymmetric on
    purpose: a small badge box should count as "worn" even though its IoU
    with a much larger person box would be tiny."""
    ix1, iy1, ix2, iy2 = inner_xyxy
    ox1, oy1, ox2, oy2 = outer_xyxy
    ax1, ay1 = max(ix1, ox1), max(iy1, oy1)
    ax2, ay2 = min(ix2, ox2), min(iy2, oy2)
    inter = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / inner_area if inner_area > 0 else 0.0


def _box_center(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _box_diag(b):
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def class_nms(dets, iou_thr, containment_thr):
    """Suppress duplicate/nested same-class boxes before they ever reach a
    tracker. Plain IoU-NMS alone misses a small box nested fully inside a
    larger one of the same physical object (a partial-badge box inside a
    whole-badge box) - their IoU stays low even though only one of them is
    real, because IoU divides by the (large) union rather than the small
    box's own area. Checks containment in both directions so it doesn't
    matter which of the two boxes happens to score higher.

    The containment check alone is not enough in a dense/crowded scene: a
    single oversized, sloppy box can fully contain a second, real, clearly
    SEPARATE detection elsewhere in the crowd (confirmed directly against
    video_inf_1/4 - see the NMS/crowd probe's events.csv), which would
    wrongly delete a correct detection instead of a duplicate. So a
    containment-only trigger additionally requires the two boxes' centers
    to be close relative to the smaller box's own size (NMS_CONTAINMENT_
    DIST_RATIO) - true nested duplicates (a badge sub-part, same-object
    jitter) are naturally co-located; a crowd false-merge is not. The IoU
    trigger is left ungated since the data showed it wasn't the culprit.

    dets: list of (bbox_xyxy, label, score). Returns the kept subset."""
    if not dets:
        return dets
    order = sorted(range(len(dets)), key=lambda i: dets[i][2], reverse=True)
    keep, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(dets[i])
        bi = dets[i][0]
        for j in order:
            if j == i or j in suppressed:
                continue
            bj = dets[j][0]
            if iou_xyxy(bi, bj) >= iou_thr:
                suppressed.add(j)
                continue
            if containment(bi, bj) >= containment_thr or containment(bj, bi) >= containment_thr:
                smaller_diag = min(_box_diag(bi), _box_diag(bj))
                center_dist = float(np.linalg.norm(_box_center(bi) - _box_center(bj)))
                if smaller_diag > 0 and center_dist <= NMS_CONTAINMENT_DIST_RATIO * smaller_diag:
                    suppressed.add(j)
    return keep


def assign_by_containment(worn_tracks, person_tracks, thr, margin=0):
    """Hungarian-match worn-item tracks (badge or coat) to person tracks by
    containment score. `margin` (default 0, i.e. no change from the original
    exact-containment check) expands the outer (person) box first - used by
    the bed-attach gate, where the attached item hangs just outside the
    anchor's own tight box rather than inside it. Returns
    {worn_track_id: person_track_id}."""
    if not worn_tracks or not person_tracks:
        return {}
    cost = np.zeros((len(worn_tracks), len(person_tracks)))
    for i, wt in enumerate(worn_tracks):
        for j, pt in enumerate(person_tracks):
            outer = expand_box(pt.last_bbox, margin)
            cost[i, j] = 1.0 - containment(wt.last_bbox, outer)
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = {}
    for r, c in zip(row_ind, col_ind):
        if 1.0 - cost[r, c] >= thr:
            assignment[worn_tracks[r].id] = person_tracks[c].id
    return assignment


def assign_sticky(worn_tracks, person_tracks, thr, bond, stale_counts, margin=0):
    """Like assign_by_containment, but the pairing persists across frames
    instead of being recomputed from a blank slate every time: once a
    worn-item track is bonded to a person, that pairing is exclusive on
    both sides for as long as both tracks stay alive - a different
    worn-item track of this class can't take that person, and this track
    can't move to a different person. Only still-unbonded tracks on both
    sides compete via Hungarian each frame. Closes the ping-pong window a
    duplicate/fragmented track or a passing occlusion could otherwise
    exploit to steal a single frame's assignment.

    `bond` ({worn_track_id: person_track_id}) is mutated in place and
    persists for the life of the video. A bond dissolves when either
    side's track dies (ages out / fails is_displayable), freeing both to
    re-bond - this is track-lifetime sticky, not a permanent decision
    like the role lock.

    It also self-corrects: a bond formed under bad conditions (e.g. only
    one person track existed yet, so it "won" by default) can otherwise
    survive forever even after its containment collapses to 0 and a
    clearly correct candidate appears - confirmed directly on real
    footage (video_inf_1: a bonded pair's containment stayed at exactly
    0.0 for 40+ consecutive frames while a different, unbonded person sat
    at ~1.0 containment the whole time, and the wrong bond never let go
    on its own). `stale_counts` ({worn_track_id: consecutive frame count
    below thr}) tracks this per bond; once a bond's CURRENT containment
    stays below `thr` for BOND_RELEASE_GRACE consecutive frames, it's
    released back to the free pool. The grace window is what keeps this
    from breaking a bond on a single occlusion-blip frame, which is the
    entire reason bonds are sticky in the first place."""
    worn_by_id = {t.id: t for t in worn_tracks}
    person_by_id = {t.id: t for t in person_tracks}

    for wid in list(bond.keys()):
        pid = bond[wid]
        if wid not in worn_by_id or pid not in person_by_id:
            del bond[wid]
            stale_counts.pop(wid, None)
            continue
        cur = containment(worn_by_id[wid].last_bbox, expand_box(person_by_id[pid].last_bbox, margin))
        if cur < thr:
            stale_counts[wid] = stale_counts.get(wid, 0) + 1
            if stale_counts[wid] >= BOND_RELEASE_GRACE:
                del bond[wid]
                stale_counts.pop(wid, None)
        else:
            stale_counts[wid] = 0

    bonded_person_ids = set(bond.values())
    free_worn = [t for t in worn_tracks if t.id not in bond]
    free_person = [t for t in person_tracks if t.id not in bonded_person_ids]

    bond.update(assign_by_containment(free_worn, free_person, thr, margin))
    return dict(bond)


def coat_extent_ratio(coat_bbox, person_bbox):
    """(coat_bbox_bottom - person_bbox_top) / person_bbox_height. None if
    the person box is too small to trust (see MIN_PERSON_H_FOR_EXTENT)."""
    person_h = person_bbox[3] - person_bbox[1]
    if person_h < MIN_PERSON_H_FOR_EXTENT:
        return None
    return (coat_bbox[3] - person_bbox[1]) / person_h


def get_role(person_track):
    coat_hits = getattr(person_track, "coat_hits", None)
    badge_hits = getattr(person_track, "badge_hits", None)
    coat_rate = (sum(coat_hits) / len(coat_hits)) if coat_hits else 0.0
    badge_rate = (sum(badge_hits) / len(badge_hits)) if badge_hits else 0.0
    if coat_rate >= COAT_ROLE_THR:
        return "Doctor"
    if badge_rate >= BADGE_ROLE_THR:
        return "Caregiver"
    return "Person"


def bed_track_ok(t):
    """Whether a 'bed' track is proven enough to gate on. Deliberately NOT
    Track.is_displayable() - that method reads the module-level PROOF_THR/
    HIT_RATE_THR from tracker_inference.py, calibrated for the fine-tuned
    classes' much stronger signal. 'bed' is an untrained open-vocab phrase
    riding on the base model's weaker general alignment (see BED_PROOF_THR)."""
    return t.confirmed and t.hit_rate() >= BED_HIT_RATE_THR and t.peak_score >= BED_PROOF_THR


def stable_urine_label(t):
    """Hysteresis wrapper around Track.smoothed_label(): once a label is
    locked in for this track, a different label must dominate
    STATE_SWITCH_RATE_THR of the whole STATE_HOLD_WINDOW (not just win a
    short-window plurality) before the displayed label is allowed to
    change. State, not the Track class, carries the ad hoc attributes -
    same pattern as pt.badge_hits/coat_hits above."""
    if not hasattr(t, "locked_label"):
        t.locked_label = None
        t.switch_history = deque(maxlen=STATE_HOLD_WINDOW)
    current = t.smoothed_label()
    if t.locked_label is None:
        t.locked_label = current
        return t.locked_label
    t.switch_history.append(current)
    if len(t.switch_history) == STATE_HOLD_WINDOW:
        counts = Counter(t.switch_history)
        best_label, best_count = counts.most_common(1)[0]
        if best_label != t.locked_label and best_count / STATE_HOLD_WINDOW >= STATE_SWITCH_RATE_THR:
            t.locked_label = best_label
    return t.locked_label


def run_tracker_on_video(model, in_path, out_path, score_thr, low_thr):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # urine-bag pipeline: one class-agnostic tracker, exactly as tracker_inference.py
    urine_tracker = ByteTracker()
    urine_class_hits = {c: 0 for c in URINE_TARGET_CLASSES}

    # role pipeline: three independent, class-specific trackers - never touch
    # or get touched by urine_tracker
    person_tracker = ByteTracker()
    badge_tracker = ByteTracker()
    coat_tracker = ByteTracker()
    role_seen = {}

    # bed anchor (v4): its own tracker, never drawn, only used to gate which
    # urine-bag tracks are allowed to display
    bed_tracker = ByteTracker()
    bag_bed_bond = {}    # urine_track_id -> bed_track_id, persists while both alive
    bag_bed_stale = {}   # urine_track_id -> consecutive frames below BED_ATTACH_THR

    # sticky cross-frame state for the role pipeline (all keyed by track id,
    # never stored as Track attributes, so it survives an appearance re-ID
    # revival that recreates the underlying Track object under the same id)
    badge_bond = {}     # badge_track_id -> person_track_id, persists while both alive
    coat_bond = {}       # coat_track_id -> person_track_id, persists while both alive
    badge_stale = {}     # badge_track_id -> consecutive frames below CONTAINMENT_THR (decay-release)
    coat_stale = {}       # coat_track_id -> consecutive frames below CONTAINMENT_THR (decay-release)
    role_streak = {}     # person_id -> (role, consecutive_frame_count) while still unlocked
    locked_role = {}     # person_id -> permanently locked role, once confirmed

    def display_role(pt):
        return locked_role.get(pt.id) or get_role(pt)

    frame_idx = 0
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

        # --- urine-bag pipeline: identical logic to tracker_inference.py ---
        distractor_mask = (labels >= URINE_TARGET_END) & (labels < URINE_DISTRACTOR_END) & (scores >= DISTRACTOR_VETO_THR)
        distractor_boxes = bboxes[distractor_mask]

        urine_target_mask = labels < URINE_TARGET_END
        urine_high_mask = urine_target_mask & (scores >= score_thr)
        urine_low_mask = urine_target_mask & (scores >= low_thr) & (scores < score_thr)

        def not_vetoed(i):
            return not any(boxes_near(db, bboxes[i], VETO_MARGIN) for db in distractor_boxes)

        urine_high_dets = [(bboxes[i], CLASS_NAMES[labels[i]], scores[i])
                            for i in np.where(urine_high_mask)[0] if not_vetoed(i)]
        urine_low_dets = [(bboxes[i], CLASS_NAMES[labels[i]], scores[i])
                           for i in np.where(urine_low_mask)[0] if not_vetoed(i)]
        urine_tracker.step(frame, urine_high_dets, urine_low_dets)

        # --- bed anchor (v4): tracked but never drawn, gates urine-bag display below ---
        bed_mask = labels >= STAFF_END
        bed_dets_raw = [(bboxes[i], "bed", scores[i]) for i in np.where(bed_mask)[0] if scores[i] >= low_thr]
        bed_dets_nms = class_nms(bed_dets_raw, NMS_IOU_THR, NMS_CONTAINMENT_THR)
        bed_high = [d for d in bed_dets_nms if d[2] >= score_thr]
        bed_low = [d for d in bed_dets_nms if d[2] < score_thr]
        bed_tracker.step(frame, bed_high, bed_low)

        # --- role pipeline: person / badge / white coat, fully separate ---
        staff_mask = (labels >= URINE_DISTRACTOR_END) & (labels < STAFF_END)
        by_class_raw = {c: [] for c in STAFF_CLASSES}
        for i in np.where(staff_mask)[0]:
            if scores[i] < low_thr:
                continue
            cls_name = CLASS_NAMES[labels[i]]
            by_class_raw[cls_name].append((bboxes[i], cls_name, scores[i]))

        by_class = {c: {"high": [], "low": []} for c in STAFF_CLASSES}
        for c in STAFF_CLASSES:
            for bbox, cls_name, score in class_nms(by_class_raw[c], NMS_IOU_THR, NMS_CONTAINMENT_THR):
                bucket = "high" if score >= score_thr else "low"
                by_class[c][bucket].append((bbox, cls_name, score))

        person_tracker.step(frame, by_class["person"]["high"], by_class["person"]["low"])
        badge_tracker.step(frame, by_class["badge"]["high"], by_class["badge"]["low"])
        coat_tracker.step(frame, by_class["white coat"]["high"], by_class["white coat"]["low"])

        person_tracks = [t for t in person_tracker.tracks if t.is_displayable()]
        badge_tracks = [t for t in badge_tracker.tracks if t.is_displayable()]
        coat_tracks = [t for t in coat_tracker.tracks if t.is_displayable()]

        # a person whose role is already permanently locked has no more use for
        # coat/badge evidence - excluding them here stops them ever taking (or
        # continuing to hold) a bond that a still-unresolved person needs. This
        # also makes assign_sticky's own stale-bond cleanup release anything
        # they were already holding (they're simply absent from person_by_id).
        competing_person_tracks = [pt for pt in person_tracks if pt.id not in locked_role]
        badge_assignment = assign_sticky(badge_tracks, competing_person_tracks, CONTAINMENT_THR, badge_bond, badge_stale)
        coat_assignment = assign_sticky(coat_tracks, competing_person_tracks, CONTAINMENT_THR, coat_bond, coat_stale)
        assigned_ids_badge = set(badge_assignment.values())
        assigned_ids_coat = set(coat_assignment.values())
        coat_box_by_id = {t.id: t.last_bbox for t in coat_tracks}
        coat_bbox_of_person = {pid: coat_box_by_id[wid] for wid, pid in coat_assignment.items() if wid in coat_box_by_id}

        for pt in person_tracks:
            if not hasattr(pt, "badge_hits"):
                pt.badge_hits = deque(maxlen=ROLE_EVIDENCE_WINDOW)
                pt.coat_hits = deque(maxlen=ROLE_EVIDENCE_WINDOW)
                pt.extent_hits = deque(maxlen=ROLE_EVIDENCE_WINDOW)
                pt.shirt_flagged = False
            pt.badge_hits.append(pt.id in assigned_ids_badge)

            coat_this_frame = pt.id in assigned_ids_coat
            extent_veto = False
            if coat_this_frame:
                extent = coat_extent_ratio(coat_bbox_of_person[pt.id], pt.last_bbox)
                if extent is not None:
                    pt.extent_hits.append(extent < COAT_EXTENT_THR)
                    extent_rate = sum(pt.extent_hits) / len(pt.extent_hits)
                    extent_veto = extent_rate >= EXTENT_VETO_RATE_THR
            pt.coat_hits.append(coat_this_frame and not extent_veto)
            # sticky, not per-frame: once this person's coat evidence has
            # been vetoed, their coat box stays hidden for good, even after
            # a Caregiver lock drops them out of coat_assignment (assign_sticky
            # excludes locked people from the competing pool, which dissolves
            # their bond - the box must not reappear just because the bond did).
            if extent_veto:
                pt.shirt_flagged = True

            if pt.id not in locked_role:
                current = get_role(pt)
                if current == "Person":
                    role_streak[pt.id] = None
                else:
                    prev = role_streak.get(pt.id)
                    streak = prev[1] + 1 if prev is not None and prev[0] == current else 1
                    role_streak[pt.id] = (current, streak)
                    if streak >= ROLE_CONFIRM_FRAMES:
                        locked_role[pt.id] = current

            role_seen.setdefault(pt.id, set()).add(display_role(pt))

        # --- bed-anchor gate (v4): which urine-bag tracks are even allowed to display ---
        displayable_bed_tracks = [t for t in bed_tracker.tracks if bed_track_ok(t)]
        bed_present = len(displayable_bed_tracks) > 0
        urine_displayable = [t for t in urine_tracker.tracks if t.is_displayable()]
        bag_bed_assignment = assign_sticky(urine_displayable, displayable_bed_tracks, BED_ATTACH_THR,
                                            bag_bed_bond, bag_bed_stale, margin=BED_ATTACH_MARGIN)
        attached_bag_ids = set(bag_bed_assignment.keys())

        # --- draw both pipelines onto the same frame ---
        for t in urine_displayable:
            if not bed_present or t.id not in attached_bag_ids:
                continue  # no bed tracked this frame, or not the bag attached to it - false positive
            label = stable_urine_label(t)
            color = URINE_PALETTE[label]
            x1, y1, x2, y2 = t.last_bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            tag = " [rec]" if t.recovered_last else ""
            text = f"#{t.id} {label}: {t.last_score:.2f}{tag}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            urine_class_hits[label] += 1

        for t in badge_tracks:
            x1, y1, x2, y2 = [int(v) for v in t.last_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROLE_PALETTE["badge"], 1)
        for t in coat_tracks:
            owner, best_c = None, 0.0
            for pt in person_tracks:
                c = containment(t.last_bbox, pt.last_bbox)
                if c > best_c:
                    owner, best_c = pt, c
            if owner is not None and best_c >= CONTAINMENT_THR and owner.shirt_flagged:
                continue
            x1, y1, x2, y2 = [int(v) for v in t.last_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROLE_PALETTE["white coat"], 1)
        for t in person_tracks:
            role = display_role(t)
            color = ROLE_PALETTE[role]
            x1, y1, x2, y2 = [int(v) for v in t.last_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"#{t.id} {role}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1 + 3, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

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
    return frame_idx, urine_class_hits, role_seen, locked_role


@contextlib.contextmanager
def _trusted_checkpoint_load():
    """PyTorch >=2.6 defaults torch.load(weights_only=True), which rejects this
    checkpoint (it isn't pure tensors). Scoped to exactly the one init_detector()
    call below that loads our own trusted weights/model.pth - never apply
    weights_only=False more broadly than that, since it re-enables arbitrary code
    execution on whatever gets loaded."""
    original = torch.load
    torch.load = lambda *a, **kw: original(*a, **{**kw, "weights_only": False})
    try:
        yield
    finally:
        torch.load = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--low_thr", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    with _trusted_checkpoint_load():
        model = init_detector(args.config, args.checkpoint, device=args.device)

    videos = sorted(f for f in os.listdir(args.video_dir) if f.lower().endswith(".mp4"))
    print(f"Found {len(videos)} videos: {videos}", flush=True)

    for v in videos:
        Track._next_id = 1  # reset IDs per video, same convention as tracker_inference.py
        in_path = os.path.join(args.video_dir, v)
        out_path = os.path.join(args.out_dir, f"tracked_{v}")
        print(f"Processing {v} -> {out_path}", flush=True)
        n_frames, urine_class_hits, role_seen, locked_role = run_tracker_on_video(
            model, in_path, out_path, args.score_thr, args.low_thr)

        print(f"\n=== {v} ===")
        print(f"  {n_frames} frames")
        for c, cnt in urine_class_hits.items():
            if cnt > 0:
                print(f"    {c}: {cnt}/{n_frames} frames ({100*cnt/n_frames:.1f}%)")
        n_doctor = sum(1 for r in locked_role.values() if r == "Doctor")
        n_caregiver = sum(1 for r in locked_role.values() if r == "Caregiver")
        n_unlocked = len(role_seen) - len(locked_role)
        print(f"    {len(role_seen)} distinct person tracks "
              f"({n_doctor} locked Doctor, {n_caregiver} locked Caregiver, {n_unlocked} never locked)")


if __name__ == "__main__":
    main()
