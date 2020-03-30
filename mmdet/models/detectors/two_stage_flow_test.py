#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 下午2:19
# @Author  : FengDa
# @File    : two_stage_flow_test.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
import mmcv
import torchvision.models as models
import torch.nn.functional as F

from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.core import bbox_overlaps, bbox2result_with_id
# from ..flow_heads.propagation_utils import warp
from ..flow_heads.propagation_utils_deprecated import warp
# from ..decision_net.utils import resize, change_img_meta, trans_action
from mmdet.models.flow_heads import FlowNetC, flownetc


@DETECTORS.register_module
class TwoStageDetectorFlowTest(BaseDetector, RPNTestMixin,
                           MaskTestMixin):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 track_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 flow_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        :param backbone: ConfigDict,采用的backbone网络结构
        :param neck: ConfigDict,采用的neck网络结构
        :param rpn_head: ConfigDict,采用的rep_head网络结构
        :param bbox_roi_extractor: ConfigDict,采用的bbox_roi_extractor网络结构
        :param bbox_head: ConfigDict,采用的bbox_head网络结构
        :param track_head: ConfigDict,采用的track_head网络结构
        :param mask_roi_extractor: ConfigDict,采用的mask_roi_extractor网络结构
        :param mask_head: ConfigDict,采用的mask_head网络结构
        :param train_cfg: ConfigDict, 训练配置参数
        :param test_cfg: ConfigDict, 测试配置参数
        :param pretrained: str, e.g.: 'modelzoo://resnet50'
        """

        super(TwoStageDetectorFlowTest, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if track_head is not None:
            self.track_head = builder.build_head(track_head)
        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)
        if flow_head is not None:
            self.flow_head = builder.build_head(flow_head)

        self.flow_head1 = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.prev_bboxes = None
        self.prev_roi_feats = None
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_flow(self):
        return self.flow_head is not None

    def load_flow(self):
        # flow_network_data = torch.load("/home/ubuntu/code/fengda/MaskTrackRCNN/pretrained_models/flownetc_EPE1.766.tar")
        # print("=> using pre-trained model '{}'".format(flow_network_data['arch']))
        # self.flow_model = FlowNetC(batchNorm=False)
        # self.flow_model.load_state_dict(flow_network_data['state_dict'])
        # # flow_model = models.__dict__[flow_network_data['arch']](flow_network_data).cuda()
        # self.flow_model.cuda()
        # self.flow_model.eval()
        # from mmdet.models.flow_heads import FlowNetC
        self.flow_head = FlowNetC(batchNorm=False,
            checkpoint="/home/ubuntu/code/fengda/MaskTrackRCNN/pretrained_models/flownetc_EPE1.766.tar")
        # self.flow_head1.requires_grad = False

    def init_weights(self, pretrained=None):
        """initialize weights."""
        super(TwoStageDetectorFlowTest, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()
        if self.with_track:
            self.track_head.init_weights()
        # if self.with_flow:
        #     self.flow_head.init_weights()
        #     self.flow_head.cuda()
        #     self.flow_head.requires_grad = False

    def extract_feat(self, img):
        """extract feature map with backbone and neck."""
        x, feat_res0 = self.backbone(img)
        # feat_resnet = x
        if self.with_neck:
            x = self.neck(x)
        return x, feat_res0

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      ref_img,  # images of reference frame
                      ref_bboxes=None,  # gt bbox of reference frame
                      gt_pids=None,  # gt ids of current frame bbox mapped to reference frame
                      gt_masks=None,
                      proposals=None,
                      key_frame=None,
                      train_flow=None,
                      key_feature_maps=None,
                      cur_feature_maps=None):
        if train_flow is not None:
            return self.train_flow(img, img_meta, gt_bboxes, gt_bboxes_ignore, gt_labels, ref_img, gt_masks,
                                   key_feature_maps=key_feature_maps, cur_feature_maps=cur_feature_maps)

        return 0, None
        is_first = img_meta[0]['is_first']
        if is_first:
            key_frame = None

        # Extract feature
        x, feat_res0 = self.extract_feat(img)
        ref_x, ref_feat_res0 = self.extract_feat(ref_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i], gt_pids[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_pids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            ref_rois = bbox2roi(ref_bboxes)
            ref_bbox_img_n = [x.size(0) for x in ref_bboxes]
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            ref_bbox_feats = self.bbox_roi_extractor(
                ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            # fetch bbox and object_id targets
            bbox_targets, (ids, id_weights) = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
            match_score = self.track_head(bbox_feats, ref_bbox_feats,
                                          bbox_img_n, ref_bbox_img_n)
            loss_match = self.track_head.loss(match_score,
                                              ids, id_weights)
            losses.update(loss_match)

        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses, x

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        is_first = img_meta[0]['is_first']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        if det_bboxes.nelement() == 0:
            det_obj_ids = np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes = None
                self.prev_roi_feats = None
                self.prev_det_labels = None
            return det_bboxes, det_labels, det_obj_ids

        res_det_bboxes = det_bboxes.clone()
        if rescale:
            res_det_bboxes[:, :4] *= scale_factor

        det_rois = bbox2roi([res_det_bboxes])
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)
        # recompute bbox match feature

        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
        else:

            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]
            match_score = self.track_head(det_roi_feats, self.prev_roi_feats,
                                          bbox_img_n, prev_bbox_img_n)[0]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:, :4], self.prev_bboxes[:, :4])
            # compute comprehensive score
            comp_scores = self.track_head.compute_comp_scores(match_logprob,
                                                              det_bboxes[:, 4].view(-1, 1),
                                                              bbox_ious,
                                                              label_delta,
                                                              add_bbox_dummy=True)
            match_likelihood, match_ids = torch.max(comp_scores, dim=1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object,
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        # udpate feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

        return det_bboxes, det_labels, det_obj_ids

    def full_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without Flow Head."""
        x, _ = self.extract_feat(img)
        key_feat_maps = x

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels,
                rescale=rescale, det_obj_ids=det_obj_ids)

            return bbox_results, segm_results, key_feat_maps

    def low_test(self, img, img_meta, key_img, key_feats, proposals=None, rescale=False):
        """Test with Flow Head."""
        full_img = key_img
        current_img = img
        feat_last = key_feats
        size_full = full_img.shape
        size_current = current_img.shape
        if size_full[-1] >= size_current[-1]:
            size = size_current[-2:]
            full_img = self.resize(full_img, size)
        else:
            size = size_full[-2:]
            current_img = self.resize(current_img, size)

        # flow
        flow_output = self.flow_head(full_img, current_img)
        upsampling = None
        if upsampling is not None:
            flow_output = F.interpolate(flow_output, size=current_img.size()[-2:], mode='bilinear', align_corners=False)
        # extract feat
        x, _ = self.extract_feat(img)
        # img_size = img.shape[-2:]
        # x = self.warp_feat(feat_last, flow_output*20, (img_size[0], img_size[1]), mode='warp')
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                   self.bbox_head.num_classes)

        def feat_fusion(x, feat_last):
            feat = []
            for i in range(len(x)):
                _size = x[i].shape[-2:]
                _tmp = torch.nn.functional.interpolate(feat_last[i], _size, mode='bilinear', align_corners=True)
                # feat.append(torch.max(x[i], _tmp))
                # feat.append((x[i] + _tmp) / 2)
                feat.append((x[i] + _tmp))

            return tuple(feat)
        img_size = img.shape[-2:]
        x = self.warp_feat(feat_last, flow_output*20, (img_size[0], img_size[1]), mode='warp')
        # x_flow = self.warp_feat(feat_last, flow_outputs1)
        # x1 = [resize(x_f, x11.shape[-2:]) for x_f, x11 in zip(x_flow, x)]
        # x1 = feat_fusion(x, x_flow)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels,
                rescale=rescale, det_obj_ids=det_obj_ids)

            return bbox_results, segm_results, None

    def simple_test(self, img, img_meta, proposals=None, rescale=False, key_frame=None):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_track, "Track head must be implemented"

        is_first = img_meta[0]['is_first']

        # if rl is None:
        if is_first:
            key_frame = None
        if key_frame is not None:
            result = self.low_test(img, img_meta, key_frame['img'], key_frame['feat_map_last'],
                                   proposals=proposals, rescale=rescale)
        else:
            result = self.full_test(img, img_meta, proposals=proposals, rescale=rescale)

        return result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def train_flow(self,
                      img,  # images of key frame
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      cur_img,  # images of current frame
                      gt_masks=None,
                      proposals=None,
                      rescale=False,
                      upsampling = None,
                      key_feature_maps=None,
                      cur_feature_maps=None):
        with torch.autograd.set_detect_anomaly(True):
            losses = dict()
            # extract feature maps of key frame
            if key_feature_maps is not None:
                x_key = key_feature_maps
            else:
                x_key, _ = self.extract_feat(img)

            img = self.resize(img, cur_img.shape[-2:])
            if cur_feature_maps is not None:
                x = cur_feature_maps
            else:
                # extract feature maps of current frame
                x, _ = self.extract_feat(cur_img)

            # detect
            proposal_list = self.simple_test_rpn(
                x, img_meta, self.test_cfg.rpn) if proposals is None else proposals
            rois = bbox2roi(proposal_list, stack=False)
            roi_feats = [self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], roi) for roi in rois]
            _results = [self.bbox_head(roi_feat) for roi_feat in roi_feats]
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            det_results = [self.bbox_head.get_det_bboxes(
                roi,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=self.test_cfg.rcnn) for (roi, (cls_score, bbox_pred)) in zip(rois, _results)]
            det_bboxes = [det_result[0] for det_result in det_results]
            det_labels = [det_result[1] for det_result in det_results]

            device = det_bboxes[0].device
            for i, det_bbox in enumerate(det_bboxes):
                if len(det_bbox) == 0:
                    det_bboxes[i] = torch.tensor([[0, 0, img_shape[1], img_shape[0], 1]]).to(device).float()
                    det_labels[i] = torch.tensor([[-1]]).to(device).float()

            # calculate flow output
            flow_output = self.flow_head(img, cur_img)
            if upsampling is not None:
                flow_output = F.interpolate(flow_output, size=cur_img.size()[-2:], mode='bilinear', align_corners=False)

            # flow_output.mean().backward()
            # feature warping
            img_size = cur_img.shape[-2:]
            x = self.warp_feat(x_key, flow_output*20, (img_size[0], img_size[1]), mode='warp')

            # mask prediction
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            scale_factor = img_meta[0]['scale_factor']
            _bboxes = (det_bboxes[:, :4] * scale_factor
                       if rescale else det_bboxes)
            mask_rois = bbox2roi(_bboxes, stack=False)
            mask_feats = [self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_roi) for mask_roi in mask_rois]
            mask_pred = [self.mask_head(mask_feat) for mask_feat in mask_feats if mask_feat.size()[0] > 0]
            # mask_pred[0].backward()

            # mask ground truths
            def assign_gt_single(det_bbox, gt_bbox, pos_iou_thr = 0.5):
                bboxes = det_bbox[:, :4]
                overlaps = bbox_overlaps(gt_bbox, bboxes)
                # for each gt, which predict best overlaps with it
                # for each gt, the max iou of all predictions
                max_overlaps, argmax_overlaps = overlaps.max(dim=1)
                num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
                assigned_gt_inds = overlaps.new_full((num_gts,), -1, dtype=torch.long)
                # assign positive: above positive IoU threshold
                pos_inds = max_overlaps >= pos_iou_thr
                assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds]
                return assigned_gt_inds

            assigned_gt_inds = [assign_gt_single(det_bbox, gt_bbox) for (det_bbox, gt_bbox) in zip(det_bboxes, gt_bboxes)]

            def mask_target_single(gt_bboxes, gt_masks, cfg):
                # get mask(28X28) for each bbox.
                mask_size = cfg.mask_size
                num_gt = gt_bboxes.size(0)
                mask_targets = []
                if num_gt > 0:
                    gt_bboxes_np = gt_bboxes.cpu().numpy()
                    # pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
                    for i in range(num_gt):
                        gt_mask = gt_masks[i]
                        bbox = gt_bboxes_np[i, :].astype(np.int32)
                        x1, y1, x2, y2 = bbox
                        w = np.maximum(x2 - x1 + 1, 1)
                        h = np.maximum(y2 - y1 + 1, 1)
                        # mask is uint8 both before and after resizing
                        target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                               (mask_size, mask_size))
                        mask_targets.append(target)
                    mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
                        gt_bboxes.device)
                else:
                    mask_targets = gt_bboxes.new_zeros((0, mask_size, mask_size))
                return mask_targets

            mask_targets = torch.cat(
                [mask_target_single(gt_bbox, gt_mask, self.train_cfg.rcnn) for (gt_bbox, gt_mask) in zip(gt_bboxes, gt_masks)])

            mask_size = self.train_cfg.rcnn.mask_size
            mask_p = []
            pos_labels = []

            # assign predicted mask and label to each mask target.
            for i in range(len(mask_pred)):
                assigned_gt_ind = assigned_gt_inds[i]
                for id in assigned_gt_ind:
                    if id == -1:
                        mask = mask_pred[i].new_zeros((1, 41, mask_size, mask_size))
                        # label = torch.tensor(0).long().to(device)
                        label = gt_labels[0].new_zeros((1,))
                    else:
                        mask = mask_pred[i][id].unsqueeze(0)
                        label = det_labels[i][id] + 1
                    mask_p.append(mask)
                    pos_labels.append(torch.tensor([label]).long().to(device))
            mask_p = torch.cat(mask_p)
            # pos_labels = torch.cat(pos_labels)
            # TODO pos_labels = gt_labels or det_labels?
            pos_labels = torch.cat(gt_labels)

            # calculate loss
            loss_mask = self.mask_head.loss(mask_p, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)
        torch.cuda.empty_cache()
        return losses

    @staticmethod
    def resize(feat_map, size=(48, 64)):
        """Resize feature map to certain size."""
        key_feature = torch.nn.functional.interpolate(feat_map, size, mode='bilinear', align_corners=False)
        return key_feature

    def warp_feat(self, feat_maps, flow, img_size, feat_stride=(4, 8, 16, 32, 64), mode='warp'):
        assert mode in ['warp', 'gather'], "Ensure mode in ['warp', 'gather']."
        if len(flow.shape) != 4:
            flow = flow.unsqueeze(0)
        sizes = [(int(img_size[0]/stride), int(img_size[1]/stride)) for stride in feat_stride]
        if feat_maps[0].shape[-2:] != sizes[0]:
            feat_maps = [self.resize(feat, size) if feat.shape[-2:] != size else feat for (feat, size) in zip(feat_maps, sizes)]
        if mode == 'warp':
            x = [warp(feat_map, flow / stride) for (feat_map, stride) in zip(feat_maps, feat_stride)]
        elif mode == 'gather':
            from mmdet.models.flow_heads.propagation_utils_deprecated import gather_nd
            flows = [self.resize(flow, size) for (size) in sizes]
            x = [gather_nd(feat_map, flow / stride) for (feat_map, flow, stride) in zip(feat_maps, flows, feat_stride)]

        return x
