# udc_tracker

A Urine Bag-Doctor-Caregiver Tracker.

A fine-tuned Grounding DINO (Swin-B) model plus a video tracking pipeline that,
from a single hospital-room camera feed, does two jobs at once:

1. **Urine-bag fill-state monitoring**: `covered` / `empty` / `full` / `half full`.
2. **Staff role tracking**: distinguishes `Doctor` (white coat) from
   `Caregiver` (badge only), tracking `person` / `badge` / `white coat`
   detections and binding them to individuals over time.

One 7-class detection model, one inference pass, one combined output video.

---

## Attribution

This is a derivative of [Intellindust-AI-Lab/FT-FSOD](https://github.com/Intellindust-AI-Lab/FT-FSOD)
("*A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning
Matters and Parallel Decoder Helps*", [arXiv:2603.28182](https://arxiv.org/abs/2603.28182)),
used here as the base fine-tuning framework (mmdetection fork + Grounding
DINO parallel-decoder training recipe). This repository keeps only what's
needed to train/run the urine-bag + staff-role joint model specifically:
the original benchmark configs, challenge subproject, and result-analysis
tooling from the upstream repo have been removed. Licensed under
[Apache-2.0](./LICENSE), same as upstream.

---

## What's in this repo

```
udc_tracker/
├── tracker_roles.py     # main entry point: run the joint tracker on video
├── tracker_inference.py       # urine-bag tracking primitives (imported by the above)
├── src_path.py                 # local paths: fill these in for your machine
├── configs_custom/
│   └── grounding_dino_swin-b_finetune_udc.py   # the training config used
├── configs_cdfsod/             # base Grounding DINO configs this one inherits from
├── mmdet/                      # mmdetection fork (training/inference framework)
├── tools/
│   ├── train.py / dist_train.sh   # training entry point
│   ├── test.py / dist_test.sh     # COCO-mAP evaluation entry point
│   └── model_converters/publish_model.py   # strips optimizer state for a clean release checkpoint
├── weights/                    # put the downloaded model.pth here (see below)
└── requirements.txt
```

## Using this from other code

`tracker_roles.py` exposes two entry points, not just the CLI:

- `run_tracker_on_video(model, in_path, out_path, score_thr, low_thr)`: the
  whole-video convenience function the CLI uses. It reads a video file
  start-to-finish, writes an annotated output video, and returns a summary.
- `StreamTracker(model, score_thr, low_thr)`: the same tracking logic,
  reusable one frame at a time. `tracker = StreamTracker(...)` builds one
  connection/video's worth of tracking state (five independent trackers -
  urine bag, person, badge, coat, bed - plus the sticky bond/role-lock
  bookkeeping); `tracker.process_frame(frame)` runs a single BGR frame
  through it and returns a plain, JSON-serializable dict of what should
  display this frame (`urine_bags`, `people`, `badges`, `coats`, each with
  track ids/labels/boxes) - no drawing, just data. `run_tracker_on_video`
  itself is built on top of this: it creates one `StreamTracker` and calls
  `process_frame()` per frame in its read loop, then draws from the
  returned dict. This is what makes a frame-at-a-time consumer (a live
  video stream, a different UI, a different storage backend) possible
  without duplicating any of the tracking logic - see "API server" below
  for a real example (a live WebSocket endpoint built on exactly this).

Two independent instances of `StreamTracker` never share state - each one
owns its own five trackers and bond dictionaries, so processing multiple
videos/streams concurrently just means constructing multiple instances (the
underlying model is not required to be an exclusive resource by this class
itself, though see the API server section for why calling `inference_detector`
concurrently across instances still needs external serialization on one GPU).

## The model

7 classes, single Grounding DINO Swin-B checkpoint:

`covered urine bag` · `empty urine bag` · `full urine bag` · `half full urine bag` · `person` · `badge` · `white coat`

Trained from the [MM-Grounding-DINO Swin-B](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth)
pretrained checkpoint (`configs_custom/grounding_dino_swin-b_finetune_udc.py`).

## Tracking design (why it's not just per-frame detection)

- **One shared tracking engine** (motion prediction + spatial matching) drives
  both the bag-state and the staff-role pipelines off a single
  `inference_detector` call per frame.
- **Proof gate + two-stage recovery**: a track only displays after one
  genuinely confident detection, then needs a minimum match rate over a
  rolling window to keep displaying: a weaker second-pass check rescues
  real objects at bad angles/partial occlusion instead of dropping them.
  Lost tracks re-identify via a visual fingerprint for up to ~60s instead of
  spawning a new track.
- **Role assignment by containment, not proximity**: a badge/coat is bound
  to the person whose box actually contains it (Hungarian matching), the
  bond persists through brief occlusion, and auto-releases if geometry
  stops supporting it for ~⅓s straight.
- **Coat-vs-shirt fix**: a plain white shirt was getting misread as a
  doctor's coat. Fixed with a geometric check (how far down the person's
  body the coat detection extends, as a fraction of height), validated on
  26 hand-checked examples with a clean separation point. One-way: it can
  only remove a wrong coat reading, never invent one.
- **Bed anchoring**: bag detections are gated on a hidden, never-displayed
  "bed" anchor class: a bag only shows if a confirmed bed is in frame and
  the bag is persistently linked to it. Kills false positives on bed-less
  clutter and duplicate boxes on the same bag. Paired with state hysteresis
  (a label must hold for several seconds before it's allowed to flip).

---

## Setup

### 1. Environment

```bash
conda create -n udc-tracker python=3.10 -y
conda activate udc-tracker

# Install PyTorch matching your CUDA version, e.g. for CUDA 12.4:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# OpenMMLab stack
pip install -U openmim
mim install "mmengine"
mim install "mmcv==2.1.0"
mim install mmdet

pip install -r requirements.txt
```

**Known environment gotchas:**
- `mmcv==2.1.0` has no prebuilt wheel for every Torch/CUDA combination (e.g.
  Torch 2.6.0 + CUDA 12.4). If `mim install "mmcv==2.1.0"` falls back to a
  source build, that's expected, not a broken install; it just takes a while
  (~15 min).
- PyTorch >=2.6 defaults `torch.load(weights_only=True)`, which will refuse to
  load this checkpoint (it isn't pure tensors). `tracker_roles.py` already
  works around this itself (a `torch.load` override scoped to just the one
  `init_detector(...)` call that loads your own trusted `weights/model.pth`).
  If you call `init_detector` from your own script instead, you'll need the
  same narrow workaround, not a blanket `weights_only=False` for the whole
  process.

**Deterministic-training patch (only needed if you plan to retrain, not for
inference):** the training config sets `deterministic=True`, which requires
patching MMEngine to accept a `warn_only` flag: see
[runner.py#L698](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L698)
and [utils.py#L48](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/utils.py#L48).

### 2. Language backbone

Grounding DINO needs a local BERT text encoder. Download `bert-base-uncased`
and `nltk_data` following [mmdetection's instructions](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/usage.md#instructions).

### 3. Download the fine-tuned weights

The trained checkpoint (`model.pth`, ~900MB) isn't stored in this git repo.
Grab it from the [Releases page](../../releases) instead:

```bash
mkdir -p weights
curl -L -o weights/model.pth \
  https://github.com/theodouk11/udc_tracker/releases/download/v1.0/model.pth
```

### 4. Set your local paths

Edit `src_path.py`:

```python
MMGDINOB_PATH = 'your_checkpoint_path/grounding_dino_swin-b_pretrain_all-f9818a7c.pth'  # only needed to retrain
JOINT_URINEBAG_PATH = 'your_dataset_path/joint_urinebag_dataset'            # only needed to retrain
```

Neither is required just to run inference with `weights/model.pth`.

---

## Running it

Point the tracker at a directory of `.mp4` videos:

```bash
python tracker_roles.py \
  --config configs_custom/grounding_dino_swin-b_finetune_udc.py \
  --checkpoint weights/model.pth \
  --video_dir /path/to/your/videos \
  --out_dir /path/to/output \
  --device cuda:0
```

Each input video gets a `tracked_<name>.mp4` in `--out_dir`, annotated with
both urine-bag state boxes and person/role boxes from the same pass.

Optional thresholds: `--score_thr` (default `0.3`, the detection confidence
floor) and `--low_thr` (default `0.05`, the weaker second-pass recovery
floor used by the two-stage tracking gate described above).

### Retraining

```bash
CONFIG="configs_custom/grounding_dino_swin-b_finetune_udc.py"
./tools/dist_train.sh "$CONFIG" 1 9994 0 --work-dir /path/to/results
./tools/dist_test.sh "$CONFIG" /path/to/results/best_coco_bbox_mAP_iter_*.pth 1 9994 0 \
  --work-dir /path/to/results --out /path/to/results/test.pkl
```

Then, to publish a clean checkpoint (strips optimizer state):

```bash
python tools/model_converters/publish_model.py \
  /path/to/results/best_coco_bbox_mAP_iter_*.pth weights/model.pth
```

---

## API server

This model is also deployable as an internet-reachable FastAPI service (a
companion project, not part of this repo's own files - built specifically
to wrap `tracker_roles.py`'s `run_tracker_on_video`/`StreamTracker` above).
Documented here since it's the primary way this model actually gets used in
practice (HUA WP5/T5.1): project partners send video for inference, and a
robot client streams live frames, rather than everyone running
`tracker_roles.py` by hand.

### Why it exists, and the one constraint that shapes everything about it

A single loaded model instance drives one GPU through `inference_detector`.
It is **not safe to call this concurrently with itself** - two simultaneous
calls would contend for the same CUDA context. On the reference deployment
hardware (an RTX 3080 Ti), a single `inference_detector` call measures
**~5.64 frames/second, ~177ms/frame** - nowhere near real-time video, and
slower than the video itself for anything but the shortest clips. Every
design decision below follows from those two facts: exactly one call to the
model at a time, and every consumer of the GPU has to share a budget far
smaller than what "live video" would suggest.

### Two ways in, one shared model

**Batch upload**. Send a whole video file, poll for status, download the
annotated result once done:
```
POST   /infer/video                    upload .mp4 (+ optional score_thr/low_thr) -> 202 {job_id, status}
GET    /infer/video/{job_id}           queued / running / done / failed
GET    /infer/video/{job_id}/result    structured summary once done
GET    /infer/video/{job_id}/download  the annotated output video
DELETE /infer/video/{job_id}           remove a job's files early
```
A video can take minutes to process (see the fps number above) - not
something an HTTP request should block on - so uploads are queued and
processed one at a time by a single background worker, never run inline
with the request.

**Live streaming**. A connected client (the robot) sends one frame at a
time and gets detections back as they're processed, over a WebSocket:
```
connect  GET /stream/video  (Authorization: Bearer <token> header)
  -> server: {"type": "ready"}
  -> client: <one binary message: an encoded image frame>
  -> server: {"type": "result", "frame_index": N, "urine_bags": [...],
              "people": [...], "badges": [...], "coats": [...]}
  -> server: {"type": "ready"}
  -> ... repeats until disconnect
```
This is an **ask-and-wait** protocol: the client is only ever allowed to
send its next frame after the server has answered the previous one and
explicitly asked for another. That single rule is what keeps the ~5.64fps
ceiling from ever being a problem to design around - the server paces the
exchange, so it can never be sent frames faster than it can process them,
and there is no frame-dropping/queueing policy needed at all.

### How both paths share one GPU without racing

A single background thread is the *only* code in the whole service allowed
to call the model. Its loop, every cycle, in this order:

1. A live frame is waiting → process it immediately, answer it, check again.
2. No live frame → advance the in-progress batch video by exactly **one**
   frame, then check again.
3. No live frame, no batch video in progress → start the next queued batch
   job.
4. Nothing to do → wait briefly.

This is frame-granular, not job-granular, on purpose: a live connection is
never left waiting behind an entire in-progress batch video (which could be
tens of minutes) - only ever behind at most one batch frame's processing
time (~177ms). Measured directly: a live frame got answered in **256ms**
while a 400-frame batch job was genuinely mid-processing, instead of the
tens of seconds to minutes plain first-come-first-served queueing would
have cost. The trade-off runs the other way too, by design: while a live
connection stays continuously active, batch jobs can make near-zero
progress for as long as that connection lasts - live traffic is meant to
win.

### Using it

Every endpoint requires `Authorization: Bearer <token>` - there is no
unauthenticated access, and the service refuses to start without at least
one token configured (each token can carry a partner label,
`label:token`, for attributing requests). It's meant to run behind HTTPS
(a reverse proxy is the recommended way to terminate TLS in front of it).

To send it a video for inference: `POST` the file as multipart form data to
`/infer/video`, poll `/infer/video/{job_id}` until `status` is `done`, then
`GET /infer/video/{job_id}/result` for the structured summary (per-class bag
detections, how many people locked in as Doctor/Caregiver) and
`/infer/video/{job_id}/download` for the annotated video itself.

To stream live frames: open a WebSocket to `/stream/video` with the
`Authorization` header set, wait for `{"type": "ready"}`, send one
JPEG/PNG-encoded frame as a binary message, read back the `{"type":
"result", ...}` detections, and repeat only after each `"ready"` - never
send a second frame before the previous one has been answered.

---

## License

Apache-2.0: see [LICENSE](./LICENSE). Derivative of Intellindust-AI-Lab/FT-FSOD;
see [Attribution](#attribution) above.
