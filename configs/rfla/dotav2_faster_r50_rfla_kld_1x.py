'''
re-implemeneted results Oct. 2022
+--------------------+-------+-------------------+-------+------------------+-------+
| category           | AP    | category          | AP    | category         | AP    |
+--------------------+-------+-------------------+-------+------------------+-------+
| plane              | 0.613 | baseball-diamond  | 0.405 | bridge           | 0.224 |
| ground-track-field | 0.494 | small-vehicle     | 0.320 | large-vehicle    | 0.540 |
| ship               | 0.576 | tennis-court      | 0.786 | basketball-court | 0.391 |
| storage-tank       | 0.435 | soccer-ball-field | 0.406 | roundabout       | 0.393 |
| harbor             | 0.390 | swimming-pool     | 0.265 | helicopter       | 0.258 |
| container-crane    | 0.021 | airport           | 0.160 | helipad          | 0.000 |
+--------------------+-------+-------------------+-------+------------------+-------+
2022-10-06 16:36:14,355 - mmdet - INFO - Exp name: dotav2_faster_r50_rfla_kld_1x.py
2022-10-06 16:36:14,363 - mmdet - INFO - Epoch(val) [12][4257]	bbox_mAP: 0.3710, bbox_mAP_50: 0.6270, bbox_mAP_75: 0.3850, bbox_mAP_vt: 0.0210, bbox_mAP_t: 0.1020, bbox_mAP_s: 0.3030, bbox_mAP_m: 0.4260, bbox_mAP_copypaste: 0.371 -1.000 0.627 0.385 0.021 0.102
Loading and preparing results...
DONE (t=2.08s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=816.55s).
Accumulating evaluation results...
DONE (t=8.82s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=3000 ] = 0.371
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=3000 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=3000 ] = 0.627
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=3000 ] = 0.385
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=3000 ] = 0.021
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=3000 ] = 0.102
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=3000 ] = 0.303
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=3000 ] = 0.426
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.468
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=3000 ] = 0.473
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=3000 ] = 0.048
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=3000 ] = 0.184
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=3000 ] = 0.422
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=3000 ] = 0.529
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=3000 ] = 0.676
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=3000 ] = 0.205
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=3000 ] = 0.217
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=3000 ] = 0.418
# Class-specific LRP-Optimal Thresholds # 
 [0.906 0.837 0.882 0.801 0.685 0.746 0.751 0.93  0.863 0.784 0.872 0.85
 0.767 0.811 0.826 0.533 0.869   nan]
'''
_base_ = [
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  #Down sampled by 4, 8, 16, 32 times respectively
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='RFGenerator', # Effective Receptive Field as prior
            fpn_layer='p2', # start FPN level P2
            fraction=0.5, # the fraction of ERF to TRF
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=18,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='HieAssigner',
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl',
                topk=[2,1],
                ratio=0.9), 
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=512),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

optimizer = dict(type='SGD', lr=0.02/4, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12, metric='bbox', proposal_nums=(300, 1000, 3000))