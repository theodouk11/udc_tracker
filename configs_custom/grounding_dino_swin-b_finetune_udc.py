_base_ = '../configs_cdfsod/grounding_dino_swin-b_pretrain_all.py'
import os
from src_path import JOINT_URINEBAG_PATH, MMGDINOB_PATH

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
randomness = dict(
    seed=42,
    deterministic=True,
    diff_rank_seed=False,
    warn_only=True
)

# Joint 7-class model: the 4 urine-bag fill-state classes plus 3 classes
# from a separate hospital-staff-attire task (person / badge / white coat).
# Trained from the base pretrained checkpoint (not continued from the
# urine-bag-only checkpoint) on the union of both datasets, per explicit
# decision to keep this as one deployable model rather than two. Dataset
# built by symlinking both source datasets' train/test splits into one
# root (JOINT_URINEBAG_PATH) with a single merged, globally-consistent
# 7-category annotation file for each split - no image bytes duplicated.
# No per-video test/val split needed for the staff-attire portion since
# photo-mAP has already been shown project-wide to not predict real
# (video) performance; that val split exists only to keep this config's
# val-driven LR scheduler and save_best checkpoint selection mechanically
# functional, not as a trusted accuracy signal - that judgment comes from
# watching real inference video after training.
class_name = 'udc'
data_root = JOINT_URINEBAG_PATH

class_name_list = [
    "covered urine bag",
    "empty urine bag",
    "full urine bag",
    "half full urine bag",
    "person",
    "badge",
    "white coat",
]

class_names = tuple(class_name_list)
num_classes = len(class_names)
metainfo = dict(
    classes=class_names,
    palette=[(220, 20, 60), (60, 180, 75), (255, 165, 0), (0, 130, 200),
             (145, 30, 180), (128, 128, 0), (0, 128, 128)])

model = dict(
    type='GroundingDINO_ParallelDecoder_15_DNQuery_rand',
    rand_dnquery_rate=0.5,
    bbox_head=dict(
        type='GroundingDINOHead_ParallelDecoder_DN',
        num_classes=num_classes))

# Motion blur / JPEG compression / defocus blur + wide HSV jitter.
# hue_shift_limit=90 was specifically tuned to break a yellow/blue liquid
# colour shortcut in urine-bag training photos - it measurably helped
# `covered` but measurably hurt `empty`/`half full` in isolation, a real
# trade-off with no equivalent justification for badge/white-coat colour,
# kept anyway because deployment video is blue-dominant (see weights/model.pth's
# training notes in the project history).
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="CachedMixUp", img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=10, pad_val=(114, 114, 114), prob=0.3),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(800, 1333), (768, 1333), (960, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(600, 4200), (800, 4200), (960, 4200)],
                    keep_ratio=True),
                dict(
                    type='HumanityRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(640, 960),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(800, 1333), (768, 1333), (960, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='Albu',
        transforms=[
            dict(type='MotionBlur', blur_limit=7, p=0.3),
            dict(type='ImageCompression', quality_range=(40, 85), p=0.3),
            dict(type='GaussianBlur', blur_limit=5, p=0.2),
            dict(type='HueSaturationValue', hue_shift_limit=90, sat_shift_limit=40, val_shift_limit=30, p=0.6),
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    num_workers=4,
    batch_size=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

val_dataloader = dict(
    num_workers=4,
    batch_size=4,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        return_classes=True))

test_dataloader = dict(
    num_workers=4,
    batch_size=4,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/'),
        pipeline=test_pipeline,
        return_classes=True))

val_evaluator = dict(
    ann_file=f'{data_root}/annotations/instances_val.json',
    classwise=True)

test_evaluator = dict(
    ann_file=f'{data_root}/annotations/instances_val.json',
    classwise=True)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.2),
            'language_model': dict(lr_mult=0.2),
        }))

max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
)

auto_scale_lr = dict(base_batch_size=10)

param_scheduler = [
    dict(
        type='ReduceOnPlateauParamScheduler',
        param_name='lr',
        monitor='coco/bbox_mAP',
        rule='greater',
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_rule='rel',
        cooldown=1,
        min_value=1e-6,
        verbose=True)
]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='coco/bbox_mAP'))

custom_hooks = [
    dict(
        type='BBoxHeadFirstHook6',
        adjust_scheduler_patience=True,
        patience_frozen=3,
        patience_unfrozen=8,
    )
]

load_from = MMGDINOB_PATH
