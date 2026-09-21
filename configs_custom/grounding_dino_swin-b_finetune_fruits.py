_base_ = '../configs_cdfsod/grounding_dino_swin-b_pretrain_all.py'
import os
from src_path import FRUITS_DATASET_PATH, MMGDINOB_PATH

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
randomness = dict(
    seed=42,
    deterministic=True,
    diff_rank_seed=False,
    warn_only=True
)

# Standalone apple/tomato photo detector: a separate task from the
# urine-bag/Task5 joint model, not merged with it. Deployment target is
# still images (the held-out test split), not video, so none of the
# video-domain-gap augmentation from that project (motion blur/compression,
# and especially the wide hue jitter tuned to break a liquid-colour
# shortcut) is carried over here - colour is plausibly real signal for
# telling apples from tomatoes, not a confound to train away.
class_name = 'fruits'
data_root = FRUITS_DATASET_PATH

class_name_list = [
    "apple",
    "tomato",
]

class_names = tuple(class_name_list)
num_classes = len(class_names)
metainfo = dict(
    classes=class_names,
    palette=[(220, 20, 60), (255, 99, 71)])

model = dict(
    type='GroundingDINO_ParallelDecoder_15_DNQuery_rand',
    rand_dnquery_rate=0.5,
    bbox_head=dict(
        type='GroundingDINOHead_ParallelDecoder_DN',
        num_classes=num_classes))

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
        ann_file='instances_train.json',
        data_prefix=dict(img='images/train/'),
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
        ann_file='instances_val.json',
        data_prefix=dict(img='images/val/'),
        pipeline=test_pipeline,
        return_classes=True))

# The real held-out evaluation (10 photos, never touched by training/val
# selection) - what tools/test.py actually reports at the end of the job.
test_dataloader = dict(
    num_workers=4,
    batch_size=4,
    persistent_workers=False,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_test.json',
        data_prefix=dict(img='images/test/'),
        pipeline=test_pipeline,
        return_classes=True))

val_evaluator = dict(
    ann_file=f'{data_root}/instances_val.json',
    classwise=True)

test_evaluator = dict(
    ann_file=f'{data_root}/instances_test.json',
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
