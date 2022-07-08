from turtle import color
import torch
import random
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mmdet.core.bbox.builder import build_assigner
from mmdet.core import anchor, build_anchor_generator
from PIL import Image
from PIL import ImageDraw
from matplotlib.ticker import FormatStrFormatter
from mmdet.core.bbox.iou_calculators import build_iou_calculator


# generate anchors
class simulate_assign:
    def __init__(self,
                anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8], #8,16,32
                     ratios=[1.0],
                     strides=[4, 8, 16, 32, 64]), #4, 8, 16, 32, 64 #8, 16, 32, 64, 128
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=512),
                iou_calculator=dict(type='BboxDistanceMetric'),
                rfassigner=dict(
                    type='HieAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=512,
                    iou_calculator=dict(type='BboxDistanceMetric'),
                    topk=[8,0],
                    ratio=1.1)):
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.assigner = build_assigner(assigner) # for anchor based 
        self.rfassigner = build_assigner(rfassigner) # for RFLA
        self.iou_calculator = build_iou_calculator(iou_calculator)



    def test_imbalance(self, num_images, mode='task1', plot=False):
        '''
        main function of testing the scale/location/ratio imbalance problem

        Args:
            num_images: num of tested images
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            [torch.Size([200, 200]), torch.Size([100, 100]), torch.Size([50, 50]), torch.Size([25, 25]), torch.Size([13, 13])]
            device (torch.device | str): Device for returned tensors
        '''
        featmap_sizes = [torch.Size([200, 200]), torch.Size([100, 100]), torch.Size([50, 50]), torch.Size([25, 25]), torch.Size([13, 13])]
        # generate synthesized anchors
        multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes, "cpu")
        # Multi-level anchors of the image, which are concatenated into a single tensor of shape (num_anchors ,4)
        flat_anchors = torch.tensor([])#.cuda()
        for item in multi_level_anchors:
            flat_anchors = torch.cat((flat_anchors, item), 0) 
        #print(flat_anchors.size()) # torch.Size([479646, 4])    
        #     
        image_size = 800
        #task1: calculate average number of assigned gt, visualize the heat map
        if mode =='task1':
            gt_bboxes = self.generate_gt(image_size, num_gts=1000,scale=12, ratio=1.0, mode='task1')
            #assign gt to anchors
            assign_result = self.assigner.assign(flat_anchors, gt_bboxes.cuda())
            gt_inds = assign_result.gt_inds
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num = counts.sum(0) # number of pos samples for each gt
            self.heatmap(pos_num, gt_bboxes, flat_anchors)
            #print(pos_num)
        #task2: scale imbalance 
        if mode =='task2':
            scale_range = 64
            ra=1.0
            gt_bboxes = self.generate_gt(image_size, num_gts=1000,scale=scale_range, ratio=ra, mode='task2') 
            #assign gt to anchors
            assign_result = self.assigner.assign(flat_anchors, gt_bboxes)
            gt_inds = assign_result.gt_inds
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num = counts.sum(0) # number of pos samples for each gt
            #self.heatmap(pos_num, gt_bboxes, flat_anchors)
            ave_num = self.stat(pos_num, gt_bboxes, anchor, scale=scale_range, ratio=ra,  mode='task2', plot=plot)
            print(ave_num)
        
        # task3: aspect ratio imbalance
        if mode =='task3':
            sc = 12
            ra=4.0
            gt_bboxes = self.generate_gt(image_size, num_gts=1000,scale=sc, ratio=ra, mode='task3') 
            #assign gt to anchors
            assign_result = self.assigner.assign(flat_anchors, gt_bboxes)
            gt_inds = assign_result.gt_inds
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num = counts.sum(0) # number of pos samples for each gt
            #self.heatmap(pos_num, gt_bboxes, flat_anchors)
            ave_num = self.stat(pos_num, gt_bboxes, anchor, scale=sc, ratio=ra,  mode='task3', plot=plot)
            print(ave_num)
        
        # task4: fcos scale imbalance
        if mode =='task4':
            scale_range = 64
            ra=1.0
            number_gt =1000
            k = 20
            
            gt_bboxes = self.generate_gt(image_size, num_gts = number_gt,scale=scale_range, ratio=ra, mode='task2') 
            #assign gt to anchors
            true_center_distance = self.iou_calculator(gt_bboxes, flat_anchors, mode='center_distance')
            center_distance = 1/(1+true_center_distance)

            gt_max_dis, gt_argmax_dis = center_distance.topk(k, dim=1, largest=True, sorted=True)
            assigned_gt_inds = center_distance.new_full((flat_anchors.size(0),), 0, dtype=torch.long)
            for i in range(number_gt):
                for j in range(k):
                    max_overlap_inds = center_distance[i,:] == gt_max_dis[i,j]
                    assigned_gt_inds[max_overlap_inds] = i + 1
            # calculate the short side of gt_box
            gt_inds = assigned_gt_inds
            gt_width = gt_bboxes[..., 2] - gt_bboxes[..., 0]
            gt_height = gt_bboxes[..., 3] - gt_bboxes[..., 1]
            gt_mside = torch.min(gt_width, gt_height)
            gt_mside = gt_mside[..., None].repeat(1, flat_anchors.size(0))
            inside_flag = true_center_distance <= (gt_mside/2)
            background = torch.zeros([1, flat_anchors.size(0)])
            inside_flag = torch.cat([background, inside_flag], dim=0)
            inside_mask = inside_flag[gt_inds, torch.arange(flat_anchors.size(0))]
            gt_inds = inside_mask * gt_inds
            gt_inds = gt_inds.long()
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num = counts.sum(0) # number of pos samples for each gt
            #self.heatmap(pos_num, gt_bboxes, flat_anchors)
            ave_num = self.stat(pos_num, gt_bboxes, anchor, scale=scale_range, ratio=ra,  mode='task2', plot=plot)
            print(ave_num)

        if mode =='task24':
            ####task2 Faster R-CNN scale imbalance
            scale_range = 64
            ra=1.0
            number_gt =1000
            k = 20
            gt_bboxes = self.generate_gt(image_size, num_gts=number_gt, scale=scale_range, ratio=ra, mode='task2') 
            #assign gt to anchors
            assign_result = self.assigner.assign(flat_anchors, gt_bboxes)
            gt_inds = assign_result.gt_inds
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t2 = counts.sum(0) # number of pos samples for each gt
            #self.heatmap(pos_num, gt_bboxes, flat_anchors)

            ####task4 FCOS scale imbalance
            true_center_distance = self.iou_calculator(gt_bboxes, flat_anchors, mode='center_distance')
            center_distance = 1/(1+true_center_distance)

            gt_max_dis, gt_argmax_dis = center_distance.topk(k, dim=1, largest=True, sorted=True)
            assigned_gt_inds = center_distance.new_full((flat_anchors.size(0),), 0, dtype=torch.long)
            for i in range(number_gt):
                for j in range(k):
                    max_overlap_inds = center_distance[i,:] == gt_max_dis[i,j]
                    assigned_gt_inds[max_overlap_inds] = i + 1
            # calculate the short side of gt_box
            gt_inds = assigned_gt_inds
            gt_width = gt_bboxes[..., 2] - gt_bboxes[..., 0]
            gt_height = gt_bboxes[..., 3] - gt_bboxes[..., 1]
            gt_mside = torch.min(gt_width, gt_height)
            gt_mside = gt_mside[..., None].repeat(1, flat_anchors.size(0))
            inside_flag = true_center_distance <= (gt_mside/2)
            background = torch.zeros([1, flat_anchors.size(0)])
            inside_flag = torch.cat([background, inside_flag], dim=0)
            inside_mask = inside_flag[gt_inds, torch.arange(flat_anchors.size(0))]
            gt_inds = inside_mask * gt_inds
            gt_inds = gt_inds.long()
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t4 = counts.sum(0) # number of pos samples for each gt
            ave_num = self.stat(pos_num_t2, pos_num_t4 , gt_bboxes, anchor, scale=scale_range, ratio=ra,  mode='task24', plot=plot)

        if mode == 'task246':
            # test the scale imbalance of FCOS Faster and RFLA
            scale_range = 64
            ra=1.0
            number_gt =1000
            k = 20
            gt_bboxes = self.generate_gt(image_size, num_gts=number_gt, scale=scale_range, ratio=ra, mode='task2') 
            #assign gt to anchors
            assign_result = self.assigner.assign(flat_anchors, gt_bboxes)
            gt_inds = assign_result.gt_inds
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t2 = counts.sum(0) # number of pos samples for each gt
            #self.heatmap(pos_num, gt_bboxes, flat_anchors)

            ####task4 FCOS scale imbalance
            true_center_distance = self.iou_calculator(gt_bboxes, flat_anchors, mode='center_distance')
            center_distance = 1/(1+true_center_distance)

            gt_max_dis, gt_argmax_dis = center_distance.topk(k, dim=1, largest=True, sorted=True)
            assigned_gt_inds = center_distance.new_full((flat_anchors.size(0),), 0, dtype=torch.long)
            for i in range(number_gt):
                for j in range(k):
                    max_overlap_inds = center_distance[i,:] == gt_max_dis[i,j]
                    assigned_gt_inds[max_overlap_inds] = i + 1
            # calculate the short side of gt_box
            gt_inds = assigned_gt_inds
            gt_width = gt_bboxes[..., 2] - gt_bboxes[..., 0]
            gt_height = gt_bboxes[..., 3] - gt_bboxes[..., 1]
            gt_mside = torch.min(gt_width, gt_height)
            gt_mside = gt_mside[..., None].repeat(1, flat_anchors.size(0))
            inside_flag = true_center_distance <= (gt_mside/2)
            background = torch.zeros([1, flat_anchors.size(0)])
            inside_flag = torch.cat([background, inside_flag], dim=0)
            inside_mask = inside_flag[gt_inds, torch.arange(flat_anchors.size(0))]
            gt_inds = inside_mask * gt_inds
            gt_inds = gt_inds.long()
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t4 = counts.sum(0) # number of pos samples for each gt

            ### task 6 RFLA scale imbalance
            rf_assign_result = self.rfassigner.assign(flat_anchors, gt_bboxes)
            rf_gt_inds = rf_assign_result.gt_inds
            rf_mask = rf_gt_inds >= 0
            rf_gt_inds = rf_gt_inds *rf_mask 
            rf_counts = F.one_hot(rf_gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t6 = rf_counts.sum(0) # number of pos samples for each gt

            ave_num = self.stat(pos_num_t2, pos_num_t4, pos_num_t6, gt_bboxes, anchor, scale=scale_range, ratio=ra,  mode='task246', plot=plot)


        if mode == 'task35':
            sc = 32
            ra=4.0
            number_gt =1000
            k = 20
            gt_bboxes = self.generate_gt(image_size, num_gts=number_gt,scale=sc, ratio=ra, mode='task3') 
            #task3 Faster r-cnn as imbalance
            #assign gt to anchors
            assign_result = self.assigner.assign(flat_anchors, gt_bboxes)
            gt_inds = assign_result.gt_inds
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t3 = counts.sum(0) # number of pos samples for each gt


            # task 5 FCOS
            true_center_distance = self.iou_calculator(gt_bboxes, flat_anchors, mode='center_distance')
            center_distance = 1/(1+true_center_distance)

            gt_max_dis, gt_argmax_dis = center_distance.topk(k, dim=1, largest=True, sorted=True)
            assigned_gt_inds = center_distance.new_full((flat_anchors.size(0),), 0, dtype=torch.long)
            for i in range(number_gt):
                for j in range(k):
                    max_overlap_inds = center_distance[i,:] == gt_max_dis[i,j]
                    assigned_gt_inds[max_overlap_inds] = i + 1
            # calculate the short side of gt_box
            gt_inds = assigned_gt_inds
            gt_width = gt_bboxes[..., 2] - gt_bboxes[..., 0]
            gt_height = gt_bboxes[..., 3] - gt_bboxes[..., 1]
            gt_mside = torch.min(gt_width, gt_height)
            gt_mside = gt_mside[..., None].repeat(1, flat_anchors.size(0))
            inside_flag = true_center_distance <= (gt_mside/2)
            background = torch.zeros([1, flat_anchors.size(0)])
            inside_flag = torch.cat([background, inside_flag], dim=0)
            inside_mask = inside_flag[gt_inds, torch.arange(flat_anchors.size(0))]
            gt_inds = inside_mask * gt_inds
            gt_inds = gt_inds.long()
            counts = F.one_hot(gt_inds, num_classes = gt_bboxes.size(0)+1)
            pos_num_t5 = counts.sum(0) # number of pos samples for each gt
            ave_num = self.stat(pos_num_t3, pos_num_t5, gt_bboxes, anchor, scale=sc, ratio=ra,  mode='task35', plot=plot)

           




        # calculate number of anchors assigned to each gt, calculate average IoU

    def generate_gt(self, image_size, num_gts=20, scale=12, ratio=1.0, mode='task1'):
        pass
        '''
        generate randomly generated ground truth boxes.

         Args:
            image_size: the size of input image
            num_gts: number of generated gts
            scale: absolute scale of gts
            ratio: width / height
            mode: task1: generate gts for verifying location imbalance, same scale, ratio and different location
                  task2: generate gts for verifying scale imbalance
                  task3: generate gts for verifying ratio imbalance
        '''
        if mode == 'task1':
            # location is random
            gt_xy = torch.rand(num_gts, 2)
            gt_xy = gt_xy * image_size
            w = scale * math.sqrt(ratio)
            h = scale / math.sqrt(ratio)
            gt_x2 = gt_xy[:, 0]+w
            gt_y2 = gt_xy[:, 1]+h
            gt_bboxes  = torch.cat((gt_xy, gt_x2[..., None]), 1)
            gt_bboxes  = torch.cat((gt_bboxes, gt_y2[..., None]), 1)

        if mode == 'task2':
            # scale and location are random
            gt_xy = torch.rand(num_gts, 2)
            gt_xy = gt_xy * image_size
            scales = torch.rand(num_gts, 1)
            scales = scales * scale    # in this task, scale serve as the scale range
            w = scales * math.sqrt(ratio)
            h = scales / math.sqrt(ratio)
            gt_x2 = gt_xy[:, 0, None]+w
            gt_y2 = gt_xy[:, 1, None]+h
            gt_bboxes  = torch.cat((gt_xy, gt_x2), 1)
            gt_bboxes  = torch.cat((gt_bboxes, gt_y2), 1)

        if mode == 'task3':
            # ratio and location are random
            gt_xy = torch.rand(num_gts, 2)
            gt_xy = gt_xy * image_size
            ratios = torch.rand(num_gts, 1)
            ratios = ratios * ratio # in this task, ratio serve as the ratio range
            w = scale * torch.sqrt(ratios)
            h = scale / torch.sqrt(ratios)
            gt_x2 = gt_xy[:, 0, None]+w
            gt_y2 = gt_xy[:, 1, None]+h
            gt_bboxes  = torch.cat((gt_xy, gt_x2), 1)
            gt_bboxes  = torch.cat((gt_bboxes, gt_y2), 1)


        return gt_bboxes



    def vis(self, boxes):
        img = Image.new('RGB', (800, 800), (255, 255, 255))
        a = ImageDraw.ImageDraw(img)  #用a来表示
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            a.rectangle(((x1, y1),(x2, y2)), fill=None, outline='red', width=1)
        img.save('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/vis1.png')
         
        return None
    
    def heatmap(self, pos_num, gt_bboxes, anchors):
        img = Image.new('RGB', (800, 800), (255, 255, 255))
        a = ImageDraw.ImageDraw(img)  #用a来表示
        print(pos_num.size(0))
        '''
        for k in range(anchors.size(0)):
            center_x = (anchors[k, 0]+anchors[k, 2])/2
            center_y = (anchors[k, 1]+anchors[k, 3])/2
            r = 1
            a.ellipse((center_x-r, center_y-r, center_x+r, center_y+r), fill=(0,0,0,0))
        '''
        for i in range(pos_num.size(0)-1):
            center_x = (gt_bboxes[i, 0]+gt_bboxes[i, 2])/2
            center_y = (gt_bboxes[i, 1]+gt_bboxes[i, 3])/2
            r = (pos_num[i+1]+1)
            r2 = int(r*1.4)
            #if 
            """
            a.text((center_x.item(), center_y.item()), str(pos_num[i+1].item()), font=None, fill="red", direction=None)
            """
            if r<=2:
                color = (38,70,83,0)
            elif 3<r<=4:
                color = (42,157,143,0) #255,101,51,0
            elif r==5:
                color = (255,183,3,0)
            elif r==6:
                color = (244,162,97,0) #255,178,15,0
            elif r>=7:
                color = (231,111,81,0)
            a.ellipse((center_x-r2, center_y-r2, center_x+r2, center_y+r2), fill=color)
        
        img.save('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/heatmap.png')    

        return None

    def stat(self, pos_num_t2, pos_num_t4, pos_num_t6, gt_bboxes, anchors, scale=64, ratio=1, mode='task1', plot=False):
        if mode == 'task1':
            pass
        if mode == 'task2':
            count1 = torch.zeros((17,1)) # for counting, 16 means 16 intervals
            count2 = torch.zeros((17,1)) # for statistics
            areas = (gt_bboxes[:,3] - gt_bboxes[:,1]) * (gt_bboxes[:,2] - gt_bboxes[:,0])
            areas = torch.sqrt(areas) # absolute scale
            for i in range(areas.size(0)):
                area = areas[i]
                index = int(area // (scale/16))
                count1[index] += 1
                count2[index] += pos_num_t2[i+1]
            ave_num_t2 = count2 / count1 
            if plot == True:
                axis_x = np.linspace(0, scale, 17).astype(int)
                axis_y_t2 = ave_num_t2
                bar_width = 2.4
                index_x= np.arange(len(axis_x))
                plt.figure(figsize=(8,6))
                plt.bar(axis_x, height=axis_y_t2, width=bar_width, color='#8CB369')
                plt.yticks(np.linspace(0, 10.0, 11))
                plt.xticks(axis_x-(bar_width-0.2),axis_x)
                plt.xlabel('gt scale', fontsize = 18)
                plt.ylabel('Number of positive samples', fontsize = 18)
                plt.tick_params(labelsize=16)
                plt.legend()
                plt.savefig('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/task2.png')


        if mode == 'task24':
            count1 = torch.zeros((17,1)) # for counting, 16 means 16 intervals
            count2 = torch.zeros((17,1)) # for statistics
            areas = (gt_bboxes[:,3] - gt_bboxes[:,1]) * (gt_bboxes[:,2] - gt_bboxes[:,0])
            areas = torch.sqrt(areas) # absolute scale
            for i in range(areas.size(0)):
                area = areas[i]
                index = int(area // (scale/16))
                count1[index] += 1
                count2[index] += pos_num_t2[i+1]
            ave_num_t2 = count2 / count1 

            count1 = torch.zeros((17,1)) # for counting, 16 means 16 intervals
            count2 = torch.zeros((17,1)) # for statistics
            for i in range(areas.size(0)):
                area = areas[i]
                index = int(area // (scale/16))
                count1[index] += 1
                count2[index] += pos_num_t4[i+1]
            ave_num_t4 = count2 / count1 
            
            if plot == True:
                axis_x = np.linspace(0, scale, 17).astype(int)
                axis_y_t2 = ave_num_t2
                axis_y_t4 = ave_num_t4
                bar_width = 1.5
                index_x= np.arange(len(axis_x))
                plt.figure(figsize=(8,6))
                plt.bar(axis_x, height=axis_y_t2, width=bar_width, color='#8CB369', label='Faster R-CNN')
                plt.bar(axis_x+bar_width, height=axis_y_t4, width=bar_width, color='#6D597A', label='FCOS')
                plt.yticks(np.linspace(0, 20.0, 21))
                plt.xticks(axis_x-(bar_width-0.2),axis_x)
                plt.xlabel('gt scale', fontsize = 20)
                plt.ylabel('Number of positive samples', fontsize = 20)
                plt.tick_params(labelsize=16)
                plt.legend(loc='upper left', fontsize = 16)
                plt.savefig('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/task24.png')

        if mode == 'task246':
            count1 = torch.zeros((17,1)) # for counting, 16 means 16 intervals
            count2 = torch.zeros((17,1)) # for statistics
            areas = (gt_bboxes[:,3] - gt_bboxes[:,1]) * (gt_bboxes[:,2] - gt_bboxes[:,0])
            areas = torch.sqrt(areas) # absolute scale
            for i in range(areas.size(0)):
                area = areas[i]
                index = int(area // (scale/16))
                count1[index] += 1
                count2[index] += pos_num_t2[i+1]
            ave_num_t2 = count2 / count1 

            count1 = torch.zeros((17,1)) # for counting, 16 means 16 intervals
            count2 = torch.zeros((17,1)) # for statistics
            for i in range(areas.size(0)):
                area = areas[i]
                index = int(area // (scale/16))
                count1[index] += 1
                count2[index] += pos_num_t4[i+1]
            ave_num_t4 = count2 / count1 

            count1 = torch.zeros((17,1)) # for counting, 16 means 16 intervals
            count2 = torch.zeros((17,1)) # for statistics
            for i in range(areas.size(0)):
                area = areas[i]
                index = int(area // (scale/16))
                count1[index] += 1
                count2[index] += pos_num_t6[i+1]
            ave_num_t6 = count2 / count1 
            
            if plot == True:
                axis_x = np.linspace(0, scale, 17).astype(int)
                axis_y_t2 = ave_num_t2/2
                axis_y_t4 = ave_num_t4/2
                axis_y_t6 = ave_num_t6
                bar_width = 1.5
                index_x= np.arange(len(axis_x))
                plt.figure(figsize=(8,6))
                plt.bar(axis_x, height=axis_y_t2, width=bar_width/2, color='#8CB369', label='Faster R-CNN')
                plt.bar(axis_x+bar_width/2, height=axis_y_t4, width=bar_width/2, color='#6D597A', label='FCOS')
                plt.bar(axis_x+bar_width, height=axis_y_t6, width=bar_width/2, color='#f3722c', label='RFLA')
                plt.yticks(np.linspace(0, 10.0, 11))
                plt.xticks(axis_x-(bar_width-0.2),axis_x)
                plt.xlabel('gt scale', fontsize = 20)
                plt.ylabel('Number of positive samples', fontsize = 20)
                plt.tick_params(labelsize=16)
                plt.legend(loc='upper left', fontsize = 16)
                plt.savefig('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/task246.png')
            



        if mode == 'task3':
            count1 = torch.zeros((17,1)) # for counting
            count2 = torch.zeros((17,1)) # for statistics
            ratios = (gt_bboxes[:,2] - gt_bboxes[:,0]) / (gt_bboxes[:,3] - gt_bboxes[:,1]) 
            for i in range(ratios.size(0)):
                ra = ratios[i]
                index = int(ra // (ratio/16))
                count1[index] += 1
                count2[index] += pos_num_t2[i+1]
            ave_num = count2 / count1

            if plot == True:
                axis_x = np.linspace(0, ratio, 17)
                axis_x_str = ['0.0', ' ','0.5',' ','1.0', ' ','1.5',' ','2.0', ' ','2.5',' ','3.0', ' ','3.5',' ','4.0']
                axis_y = ave_num
                bar_width = 0.15
                index_x= np.arange(len(axis_x))
                plt.figure(figsize=(8,6))
                plt.bar(axis_x, height=axis_y, width=bar_width, color='#6D597A')
                plt.yticks(np.linspace(0, 3.5, 8))
                plt.xticks(axis_x-(bar_width-0.02), axis_x_str)
                plt.xlabel('gt aspect ratio',fontsize = 18)
                plt.ylabel('Number of positive samples', fontsize = 18)
                plt.tick_params(labelsize=16)
                plt.savefig('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/task3.png')

        if mode =='task35':
            count1 = torch.zeros((17,1)) # for counting
            count2 = torch.zeros((17,1)) # for statistics
            ratios = (gt_bboxes[:,2] - gt_bboxes[:,0]) / (gt_bboxes[:,3] - gt_bboxes[:,1]) 
            for i in range(ratios.size(0)):
                ra = ratios[i]
                index = int(ra // (ratio/16))
                count1[index] += 1
                count2[index] += pos_num_t2[i+1]
            ave_num_t3 = count2 / count1

            count1 = torch.zeros((17,1)) # for counting
            count2 = torch.zeros((17,1)) # for statistics
            for i in range(ratios.size(0)):
                ra = ratios[i]
                index = int(ra // (ratio/16))
                count1[index] += 1
                count2[index] += pos_num_t4[i+1]
            ave_num_t5 = count2 / count1

            if plot == True:
                axis_x = np.linspace(0, ratio, 17)
                axis_x_str = ['0.0', ' ','0.5',' ','1.0', ' ','1.5',' ','2.0', ' ','2.5',' ','3.0', ' ','3.5',' ','4.0']
                axis_y_t3 = ave_num_t3
                axis_y_t5 = ave_num_t5
                bar_width = 0.15
                index_x= np.arange(len(axis_x))
                plt.figure(figsize=(8,6))
                plt.bar(axis_x-0.05, height=axis_y_t3, width=0.1, color='#8CB369', label='Faster R-CNN')
                plt.bar(axis_x+0.05, height=axis_y_t5, width=0.1, color='#6D597A', label='FCOS')
                plt.yticks(np.linspace(0, 20, 21))
                plt.xticks(axis_x-(bar_width-0.02), axis_x_str)
                plt.xlabel('gt aspect ratio',fontsize = 20)
                plt.ylabel('Number of positive samples', fontsize = 20)
                plt.tick_params(labelsize=16)
                plt.legend(loc='upper left', fontsize = 16)
                plt.savefig('/home/xc/mmdet-swd/mmdetection/mmdet/models/dense_heads/vis/task35.png')


        return None
        
        


if __name__ == '__main__':
    obj=simulate_assign()
    obj.test_imbalance(1, mode='task246', plot=True)
    '''
    boxes = obj.generate_gt(800, num_gts=500, scale=12, ratio=2.0, mode='task3')
    obj.vis(boxes)
    '''