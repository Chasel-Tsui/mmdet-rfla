import torch
import torchvision
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayer

#from torch.nn import MultiheadAttention


@HEADS.register_module()
class TransRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        self.conv1 =  torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        self.encoder_layer = TransformerEncoderLayer(d_model=256, nhead=8).cuda()
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=2).cuda()



    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)


    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, img_metas):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        #print('bbox_feat_size:{}'.format(bbox_feats.size())) #1024*256*7*7
        #1024 denotes num of boxes,
        # v02.01.02
        '''
        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        #position encoding
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2)
        #pos_emb: 1024*256*7*7
        bbox_feats_ori = bbox_feats
        bbox_feats_input = pos_emb + bbox_feats
        bbox_feats_input = torch.reshape(bbox_feats_input, (d1, d2, d3)) # 1024, 256, 49
        bbox_feats_input = bbox_feats_input.permute(2, 0, 1) # 49, 1024, 256
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d2, nhead=4)
        transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        transformer_encoder = transformer_encoder.cuda()
        bbox_feats_output = transformer_encoder(bbox_feats_input)
        bbox_feats_output = bbox_feats_output.permute(1, 2, 0)
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7
        
        # v02.01.03 : 1024, 256, 7, 7
        
        # v02.01.04 : 
        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        #position encoding
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2)
        #pos_emb: 1024*256*7*7
        bbox_feats_ori = bbox_feats
        bbox_feats_input = pos_emb + bbox_feats
        bbox_feats_input = torch.reshape(bbox_feats_input, (d1, d2, d3)) # 1024, 256, 49
        bbox_feats_input = bbox_feats_input.permute(2, 0, 1) # 49, 1024, 256
        bbox_feats_output, att_weights = self.Multiheadencoder(bbox_feats_input, d2, nheads=4, nlayers=1)
        bbox_feats_output = bbox_feats_output.permute(1, 2, 0)
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7
        print(att_weights.size())
        
         # v02.01.05: 30 epochs
        
         # v02.01.06:  batch_attention
        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        #position encoding
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2)
        #pos_emb: 1024*256*7*7
        bbox_feats_ori = bbox_feats
        bbox_feats_input = pos_emb + bbox_feats
        bbox_feats_input = torch.reshape(bbox_feats_input, (d1, d2, d3)) # 1024, 256, 49
        bbox_feats_input = bbox_feats_input.permute(0, 2, 1) # 1024, 49, 256
        bbox_feats_output, att_weights = self.Multiheadencoder(bbox_feats_input, d2, nheads=4, nlayers=1)
        bbox_feats_output = bbox_feats_output.permute(0, 2, 1)
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7
        #print(att_weights.size())

        # v02.01.07: Average Pooling and Maxpooling + instance transformer attention
        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        bbox_feats_ori = torch.reshape(bbox_feats, (d1, d2, d3)) # 1024, 256, 49
        # average pooling and max pooling
        avg_out =torch.mean(bbox_feats_ori, dim=2, keepdim=True)
        max_out, _ =torch.max(bbox_feats_ori, dim=2, keepdim=True)
        # concat
        bbox_feats_input = torch.cat([avg_out, max_out], dim=2) # 1024, 256, 2
        conv1 =  torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        bbox_feats_input = bbox_feats_input.unsqueeze(3)
        bbox_feats_input = bbox_feats_input.permute(0,2,1,3) 
        bbox_feats_input = conv1(bbox_feats_input) # 1*1 conv
        bbox_feats_input = bbox_feats_input.squeeze(-1) #1024, 1, 256
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2, mode=1)
        bbox_feats_ori = bbox_feats_ori.permute(0, 2, 1) # 1024, 49, 256
        bbox_feats_input = pos_emb + bbox_feats_input
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_input, d2, nheads=4, nlayers=1)
        att_weights = torch.sum(att_weights, dim=2).unsqueeze(-1) #1, 1024, 1
        att_weights = att_weights.permute(1,0,2)
        att_weights = att_weights.repeat(1,d3,d2)
        bbox_feats_output = bbox_feats_ori * att_weights # 1024, 49, 256 perpixel multiply to do: add, residual
        bbox_feats_output = bbox_feats_output.permute(0, 2, 1) # 1024, 256, 49
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7
        

        # v02.01.08, 9, 10, 11, 14: Average Pooling and Maxpooling + instance transformer attention
        d1 = bbox_feats.size(0)  # 1024
        d2 = bbox_feats.size(1)  # 256
        d3 = bbox_feats.size(2) * bbox_feats.size(2)
        d4 = bbox_feats.size(2)  # 7
        bbox_feats_ori = torch.reshape(bbox_feats, (d1, d2, d3))  # 1024, 256, 49
        # average pooling and max pooling
        avg_out = torch.mean(bbox_feats_ori, dim=2, keepdim=True)
        max_out, _ = torch.max(bbox_feats_ori, dim=2, keepdim=True)
        # concat
        bbox_feats_input = torch.cat([avg_out, max_out], dim=2)  # 1024, 256, 2
        conv1 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        bbox_feats_input = bbox_feats_input.unsqueeze(3)
        bbox_feats_input = bbox_feats_input.permute(0, 2, 1, 3)
        bbox_feats_input = conv1(bbox_feats_input)  # 1*1 conv
        bbox_feats_input = bbox_feats_input.squeeze(-1)  # 1024, 1, 256
        pos_emb = self.positionembedding(bbox_feats, d1, d4, d4, d2, mode=1)
        bbox_feats_ori = bbox_feats_ori.permute(0, 2, 1)  # 1024, 49, 256
        bbox_feats_input = pos_emb + bbox_feats_input
        # num_layers=2 head=8
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_input, d2, nheads=8, nlayers=1)
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_att, d2, nheads=8, nlayers=1)
        att_weights = torch.sum(att_weights, dim=2).unsqueeze(-1)  # 1, 1024, 1
        att_weights = att_weights.permute(1, 0, 2)
        att_weights = att_weights.repeat(1, d3, d2)
        bbox_feats_output = bbox_feats_ori * att_weights  # 1024, 49, 256 perpixel multiply to do: add, residual
        bbox_feats_output = bbox_feats_output.permute(0, 2, 1)  # 1024, 256, 49
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output  # 1024, 256, 7, 7
        bbox_feats=bbox_feats.type(torch.HalfTensor).cuda()
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        
         # v02.01.12: sum(Average Pooling, Maxpooling) + instance transformer attention
        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        bbox_feats_ori = torch.reshape(bbox_feats, (d1, d2, d3)) # 1024, 256, 49
        # average pooling and max pooling
        avg_out =torch.mean(bbox_feats_ori, dim=2, keepdim=True) # 1024, 256, 1
        max_out, _ =torch.max(bbox_feats_ori, dim=2, keepdim=True) # 1024, 256, 1
        # sum
        bbox_feats_input = avg_out + max_out # 1024, 256, 1
        bbox_feats_input = bbox_feats_input.permute(0,2,1)  #1024, 1, 256
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2, mode=1)
        bbox_feats_ori = bbox_feats_ori.permute(0, 2, 1) # 1024, 49, 256
        bbox_feats_input = pos_emb + bbox_feats_input
        # num_layers=2, num_heads= 4
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_input, d2, nheads=4, nlayers=1)
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_att, d2, nheads=4, nlayers=1)
        att_weights = torch.sum(att_weights, dim=2).unsqueeze(-1) #1, 1024, 1
        att_weights = att_weights.permute(1,0,2)
        att_weights = att_weights.repeat(1,d3,d2)
        bbox_feats_output = bbox_feats_ori * att_weights # 1024, 49, 256 perpixel multiply to do: add, residual
        bbox_feats_output = bbox_feats_output.permute(0, 2, 1) # 1024, 256, 49
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7
        
        # v02.01.13: 1*1 conv + instance transformer attention
        d1 = bbox_feats.size(0)  # 1024
        d2 = bbox_feats.size(1)  # 256
        d3 = bbox_feats.size(2) * bbox_feats.size(2)
        d4 = bbox_feats.size(2)  # 7
        bbox_feats_ori = torch.reshape(bbox_feats, (d1, d2, d3))  # 1024, 256, 49
        # 1*1 conv
        conv1 = torch.nn.Conv2d(in_channels=d3, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        bbox_feats_input = bbox_feats_ori.unsqueeze(3)
        bbox_feats_input = bbox_feats_input.permute(0, 2, 1, 3) #1024, 49, 256, 1
        bbox_feats_input = conv1(bbox_feats_input)  # 1*1 conv
        bbox_feats_input = bbox_feats_input.squeeze(-1)  # 1024, 1, 256
        pos_emb = self.positionembedding(bbox_feats, d1, d4, d4, d2, mode=1)
        bbox_feats_ori = bbox_feats_ori.permute(0, 2, 1)  # 1024, 49, 256
        bbox_feats_input = pos_emb + bbox_feats_input
        # num_layers=2, num_heads= 4
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_input, d2, nheads=4, nlayers=1)
        bbox_feats_att, att_weights = self.Multiheadencoder(bbox_feats_att, d2, nheads=4, nlayers=1)
        att_weights = torch.sum(att_weights, dim=2).unsqueeze(-1)  # 1, 1024, 1
        att_weights = att_weights.permute(1, 0, 2)
        att_weights = att_weights.repeat(1, d3, d2)
        bbox_feats_output = bbox_feats_ori * att_weights  # 1024, 49, 256 perpixel multiply to do: add, residual
        bbox_feats_output = bbox_feats_output.permute(0, 2, 1)  # 1024, 256, 49
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output  # 1024, 256, 7, 7
        
        # v02.01.15: Spatialv2

        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        #position encoding
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2, mode=2)
        #pos_emb: 1024*256*7*7
        bbox_feats_input = pos_emb + bbox_feats
        bbox_feats_input = torch.reshape(bbox_feats_input, (d1, d2, d3)) # 1024, 256, 49
        bbox_feats_ori = bbox_feats_input
        bbox_feats_input = bbox_feats_input.permute(2, 0, 1) # 49, 1024, 256 (L,B,N)
        bbox_feats_spatial, spatial_att_w = self.Multiheadencoder(bbox_feats_input, d2, nheads=4, nlayers=1)
        bbox_feats_spatial, spatial_att_w = self.Multiheadencoder(bbox_feats_spatial, d2, nheads=4, nlayers=1)
        spatial_att_w  = torch.sum(spatial_att_w , dim=2).unsqueeze(-1) #1024, 49, 1 (B,L,L)--> (B,L,1)
        spatial_att_w  = spatial_att_w.permute(0,2,1)
        spatial_att_w  = spatial_att_w.repeat(1,d2,1) # 1024, 256, 49
        bbox_feats_output = bbox_feats_ori * spatial_att_w 
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7
        #print(att_weights.size())
        '''

        # v02.01.22
        d1 = bbox_feats.size(0) #1024
        d2 = bbox_feats.size(1) #256
        d3 = bbox_feats.size(2)*bbox_feats.size(2)
        d4 = bbox_feats.size(2) #7
        #position encoding
        pos_emb=self.positionembedding(bbox_feats, d1, d4, d4, d2)
        #pos_emb: 1024*256*7*7
        bbox_feats_ori = bbox_feats
        bbox_feats_input = pos_emb + bbox_feats
        bbox_feats_input = torch.reshape(bbox_feats_input, (d1, d2, d3)) # 1024, 256, 49
        bbox_feats_input = bbox_feats_input.permute(2, 0, 1) # 49, 1024, 256
        #bbox_feats_input = bbox_feats_input.type(torch.HalfTensor).cuda()
        bbox_feats_output = self.transformer_encoder(bbox_feats_input)
        bbox_feats_output = bbox_feats_output.permute(1, 2, 0)
        bbox_feats_output = torch.reshape(bbox_feats_output, (d1, d2, d4, d4))
        bbox_feats = bbox_feats_output # 1024, 256, 7, 7

        '''
        #add visualization of transformer, note that batch size should be set to 1
        #way1
           
        sns.set()
        name = img_metas[0]['filename']
        ori_name = img_metas[0]['ori_filename']
        ori_shape = img_metas[0]['ori_shape'] # (w,h,3)
        img_shape = img_metas[0]['img_shape']
        print(img_shape)
        save_dir = '/home/xc/mmdet-swd/mmdetection/mmdet/models/roi_heads/roi_feats_vis'
        num_instances=bbox_feats.size(0)
        channel=bbox_feats.size(1)
        #print(num_instances)
        #conv = torch.nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        #bbox_feats_vis = conv(bbox_feats)
        bbox_feats_vis = torch.sum(bbox_feats_output, dim=1)/d2
        #feats1=conv(bbox_feats_output)
        #feats2=conv(bbox_feats_ori)

        
        #bbox_feats_vis=feats1/feats2
        #for item in range(num_instances):
        for item in range(4):
            bbox_feats_item = bbox_feats_vis[item,...]
            #normalize
            d_max = bbox_feats_item.max()
            d_min = bbox_feats_item.min()
            norm_bbox_feats = (bbox_feats_item-d_min)/(d_max-d_min)
            #init sns heatmap
            norm_bbox_feats=norm_bbox_feats.squeeze(0)
            heatmap = sns.heatmap(norm_bbox_feats.cpu().detach().numpy(), square=True, cmap='YlGnBu')
            ori_feat_indx = str(item) + '_feat_' + ori_name
            output_feat_dir = save_dir + '/' + ori_feat_indx
            fig=heatmap.get_figure()
            fig.savefig(output_feat_dir)
            fig.clear()

            # draw the corrseponding img
            image = cv2.imread(name)
            #get the upper-left and down-right point of a certain bounding box
            upleft1 = rois[item, 1]/img_shape[0]*ori_shape[0]
            upleft2 = rois[item, 2]/img_shape[1]*ori_shape[1]
            rightdown1 = rois[item, 3]/img_shape[0]*ori_shape[0]
            rightdown2 = rois[item, 4]/img_shape[1]*ori_shape[1]
            upleft=(int(upleft1),int(upleft2))
            rightdown=(int(rightdown1),int(rightdown2))
            cv2.rectangle(image, upleft, rightdown, (0,255,0), 2)
            ori_name_indx=str(item)+'_'+ori_name
            output_dir = save_dir+'/'+ori_name_indx
            cv2.imwrite(output_dir, image)
            #visualize heatmap


            #print('{}\n image:{}-{}'.format(norm_bbox_feats,name,item))
        '''      

        #bbox_feats_vis =bbox_feats


        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        #print('cls_score:{}'.format(cls_score.size())) #1024*9
        #print('bbox_pred:{}'.format(bbox_pred.size())) #1024*32

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results



    def positionembedding(self, x, b, h, w, d_model, mode=2):
        # 输入是b,c,h,w
        # tensor_list的类型是NestedTtensor，内部自动附加了mask，
        # 用于表示动态shape，是pytorch中tensor新特性https://github.com/pytorch/nestedtensor
        # input: b, h, w, d_model, mode, mode '1'or '2' is used for choosing 1-d and 2-ds
        if mode == 2:
        # 附加的mask，shape是b,h,w 全是false
            mask = torch.ones((b,h,w)).cuda()
            # 因为图像是2d的，所以位置编码也分为x,y方向
            # 1 1 1 1 ..  2 2 2 2... 3 3 3...
            y_embed = mask.cumsum(1, dtype=torch.float32)
            # 1 2 3 4 ... 1 2 3 4...
            x_embed = mask.cumsum(2, dtype=torch.float32)
            '''
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            '''

            # num_pos_feats = 128
            # 0~127 self.num_pos_feats=128,因为前面输入向量是256，编码是一半sin，一半cos
            num_pos_feats=d_model / 2
            dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)

            # 输出shape=b,h,w,128
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) #b, h, w, 128
            # 每个特征图的xy位置都编码成256的向量，其中前128是y方向编码，而128是x方向编码
        else:
            pe = torch.zeros(b, d_model).cuda()
            position = torch.arange(0, b).unsqueeze(1).cuda()
            div_term = torch.exp(torch.arange(0, d_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model)).cuda()
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pos=pe.unsqueeze(0) # 1, b, d_model 
            pos=pos.permute(1, 0, 2) #b, 1, d_model

        return pos
    '''
    # b,n=256,h,w
    def Multiheadencoder(self, src, d_model, nheads, nlayers, mask=None):
       #custimized transformer encoder
        dropout=0.1
        dim_feedforward=2048
        relu= torch.nn.ReLU().cuda()
        multi_head = torch.nn.MultiheadAttention(d_model, nheads, dropout=dropout).cuda() # initiate multiheadattention
        linear1 = torch.nn.Linear(d_model, dim_feedforward).cuda()
        linear2 = torch.nn.Linear(dim_feedforward, d_model).cuda()
        norm1 = torch.nn.LayerNorm(d_model).cuda()
        norm2 = torch.nn.LayerNorm(d_model).cuda()
        dropout = torch.nn.Dropout(p=dropout).cuda()
        src1, att_weight = multi_head(src, src, src, attn_mask=mask)
        src = src + dropout(src1)
        src = norm1(src)
        src2 = linear2(dropout(relu(linear1(src))))
        src = src + dropout(src2)
        src = norm2(src)
        return src, att_weight
    '''      



    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        #print('resbbox_size:{}'.format([res.bboxes.size() for res in sampling_results]))
        #print('roi_size:{}'.format(rois.size()))  #1024*5
        bbox_results = self._bbox_forward(x, rois,img_metas)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals
        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)
        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois, img_metas)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
