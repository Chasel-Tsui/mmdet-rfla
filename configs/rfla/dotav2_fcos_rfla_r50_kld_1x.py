'''
re-implemeneted results Oct. 2022
+--------------------+-------+-------------------+-------+------------------+-------+
| category           | AP    | category          | AP    | category         | AP    |
+--------------------+-------+-------------------+-------+------------------+-------+
| plane              | 0.589 | baseball-diamond  | 0.354 | bridge           | 0.197 |
| ground-track-field | 0.322 | small-vehicle     | 0.280 | large-vehicle    | 0.522 |
| ship               | 0.560 | tennis-court      | 0.777 | basketball-court | 0.360 |
| storage-tank       | 0.384 | soccer-ball-field | 0.292 | roundabout       | 0.367 |
| harbor             | 0.333 | swimming-pool     | 0.249 | helicopter       | 0.124 |
| container-crane    | 0.000 | airport           | 0.144 | helipad          | 0.000 |
+--------------------+-------+-------------------+-------+------------------+-------+
2022-10-06 05:19:10,174 - mmdet - INFO - Exp name: dotav2_fcos_r50_rfla_kld_1x.py
2022-10-06 05:19:10,176 - mmdet - INFO - Epoch(val) [12][4257]	bbox_mAP: 0.3250, bbox_mAP_50: 0.5580, bbox_mAP_75: 0.3330, bbox_mAP_vt: 0.0090, bbox_mAP_t: 0.0660, bbox_mAP_s: 0.2360, bbox_mAP_m: 0.3890, bbox_mAP_copypaste: 0.325 -1.000 0.558 0.333 0.009 0.066
Loading and preparing results...
DONE (t=10.86s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=1419.95s).
Accumulating evaluation results...
DONE (t=32.97s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.325
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.558
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.333
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.009
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.066
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.236
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.389
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.469
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.474
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.016
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.191
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.538
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.718
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.222
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.332
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.439
# Class-specific LRP-Optimal Thresholds # 
 [0.297 0.284 0.231 0.224 0.173 0.251 0.26  0.33  0.334 0.2   0.273 0.289
 0.26  0.259 0.215 0.124 0.212 0.076]
'''
_base_ = [
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RFLA_FCOSHead',
        norm_cfg=None,
        num_classes=18,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        conv_bias=True,
        fpn_layer = 'p3', 
        fraction = 1/2,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HieAssigner',
             ignore_iof_thr=-1,
             gpu_assign_thr=256,
             iou_calculator=dict(type='BboxDistanceMetric'),
             assign_metric='kl',
             topk=[6,1],
             ratio=0.9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=3000))
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.01/2, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)