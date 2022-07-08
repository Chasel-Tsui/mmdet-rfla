import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class RankingAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='iou',
                 topk=1,
                 inside_circle=False):
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.inside_circle = inside_circle

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes, mode=self.assign_metric)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
        '''
        if self.anchor_compensate == False:
            assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        else:
            assign_result = self.assign_wrt_overlaps_eucliean_distance(bboxes, gt_bboxes, overlaps)
        '''
        assign_result =self.assign_wrt_ranking(overlaps, gt_labels)

        """
        if self.inside_circle == 'circle':
            center_distance = self.iou_calculator(gt_bboxes, bboxes, mode = 'center_distance2')
            width_gt = gt_bboxes[...,2] - gt_bboxes[...,0]
            height_gt = gt_bboxes[...,3] - gt_bboxes[...,1]
            # scale [0, 32]^2 r=2, scale [32, 256]^2 r= 1.5, scale [256, +inf]^2 r=1 for scale normalization
            '''
            scale = width_gt * height_gt
            scale_1 = scale <= 32*32
            scale_2 = (scale> 32*32) & (scale <= 256*256)
            scale_3 = scale > 256*256
            r = [2, 1.5, 1]
            gt_circle2 = ((width_gt/2)**2 + (height_gt/2) **2) *(scale_1*r[0]**2+scale_2*r[1]**2+scale_3*r[2]**2)
            '''
            r=1.5
            gt_circle = ((width_gt/2)**2 + (height_gt/2) **2) * r * r
            inside_flag = center_distance <= gt_circle[...,None]
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[(assign_result.gt_inds-1).clamp(min=0), length]
            assign_result.gt_inds *= inside_mask
        """
        if self.inside_circle=='circle':            
            device1 = gt_bboxes.device
            eps = 1e-6
            cxy_g = (gt_bboxes[..., None, :, :2] + gt_bboxes[..., None, :, 2:]) / 2
            cxy_a = (bboxes[..., :, None, :2] + bboxes[..., :, None, 2:]) / 2

            cxy_g = cxy_g.unsqueeze(-1)
            cxy_a = cxy_a.unsqueeze(-1)

            wg = gt_bboxes[..., :, None, 2] - gt_bboxes[..., :, None, 0] + eps
            hg = gt_bboxes[..., :, None, 3] - gt_bboxes[..., :, None, 1] + eps

            inv_sigma = torch.stack((4/(wg**2), torch.zeros_like(wg),
                                     torch.zeros_like(hg), 4/(hg**2))).reshape(-1,2,2)
            gaussian = torch.exp(-0.5*(cxy_a-cxy_g).permute(0, 1, 3, 2).matmul(inv_sigma).matmul(cxy_a-cxy_g)).squeeze(-1).squeeze(-1)
            inside_flag = gaussian >= torch.exp(torch.tensor([-1.5])).to(device1)
            length = range(assign_result.gt_inds.size(0))
            inside_mask = inside_flag[length, (assign_result.gt_inds-1).clamp(min=0)]
            assign_result.gt_inds *= inside_mask

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0
        #assign wrt ranking
        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels, kindex = gt_argmax_overlaps)