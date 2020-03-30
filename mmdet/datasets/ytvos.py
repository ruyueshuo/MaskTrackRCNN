import numpy as np
import os.path as osp
import os
import random
import mmcv
from .custom import CustomDataset
from .extra_aug import ExtraAugmentation
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from pycocotools.ytvos import YTVOS
from mmcv.parallel import DataContainer as DC
from .utils import to_tensor, random_scale
import torch
import torchvision


class YTVOSDataset(CustomDataset):
    CLASSES = ('person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan',
               'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow', 'fish',
               'train', 'horse', 'turtle', 'bear', 'motorbike', 'giraffe', 'leopard',
               'fox', 'deer', 'owl', 'surfboard', 'airplane', 'truck', 'zebra', 'tiger',
               'elephant', 'snowboard', 'boat', 'shark', 'mouse', 'frog', 'eagle', 'earless_seal',
               'tennis_racket')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 every_frame=False,
                 is_flow=False,
                 flow_test=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)

        self.every_frame = every_frame
        self.is_flow = is_flow
        self.flow_test = flow_test
        if self.flow_test or self.is_flow:
            self.cuda = True
        self.cuda = False
        if self.cuda:
            from mmcv import Config
            from mmdet.models import build_detector
            from mmcv.runner import load_checkpoint
            cfg = Config.fromfile("../configs/masktrack_rcnn_r50_fpn_1x_flow_youtubevos.py")
            self.det_model = build_detector(
                cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
            load_checkpoint(self.det_model, "../results/20200312-180434/epoch_9.pth")
            self.det_model = self.det_model.cuda()
            self.det_model.eval()
            for param in self.det_model.parameters():
                param.requires_grad = False

        # Set indexes for data loading
        img_ids = []  # training frames which have annotations
        img_ids_all = []  # all training frames
        img_ids_pairs = []  # flow data pairs
        for idx, vid_info in enumerate(self.vid_infos):
            vid_name = vid_info['filenames'][0].split('/')[0]
            folder_path = osp.join(self.img_prefix, vid_name)
            files = os.listdir(folder_path)
            files.sort()
            vid_info['filenames_all'] = [osp.join(vid_name, file) for file in files]
            for _id in range(len(files)):
                img_ids_all.append((idx, _id))
                is_anno = vid_info['filenames_all'][_id] in vid_info['filenames']
                if is_anno and _id > 0:  # having annotation and is not the first frame.
                    ann_idx = vid_info['filenames'].index(vid_info['filenames_all'][_id])
                    ann = self.get_ann_info(idx, ann_idx)
                    gt_bboxes = ann['bboxes']
                    # skip the image if there is no valid gt bbox
                    if len(gt_bboxes)==0:
                        continue
                    # random select key frame
                    key_id = _id - np.random.randint(1, min(10, _id))
                    img_ids_pairs.append(((idx, key_id), (idx, _id)))
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))

        self.img_ids = img_ids
        self.img_ids_all = img_ids_all
        self.img_ids_pairs = img_ids_pairs

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                          if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

    def __len__(self):
        if self.every_frame:
            return len(self.img_ids_all)
        elif self.is_flow:
            return len(self.img_ids_pairs)
        else:
            return len(self.img_ids)

    def __getitem__(self, idx):
        if self.test_mode:
            if self.every_frame:
                return self.prepare_test_img(self.img_ids_all[idx])
            else:
                return self.prepare_test_img(self.img_ids[idx])
        if self.is_flow:
            if self.flow_test:
                data = self.prepare_train_flow_test_img(self.img_ids_pairs[idx])
            else:
                data = self.prepare_train_flow_img(self.img_ids_pairs[idx])
        else:
            data = self.prepare_train_img(self.img_ids[idx])
        return data

    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]

        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2 * np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2 * np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:, :2] + bbox[:, 2:]) / 2.
        sizes = bbox[:, 2:] - bbox[:, :2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes / 2.
        new_x2y2 = new_centers + new_sizes / 2.
        c_min = [0, 0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1, new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
            # check if the frame id is valid
            ref_idx = (vid, i)
            if i != frame_id and ref_idx in self.img_ids:
                valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def prepare_train_flow_test_img(self, idx):

        # prepare a pair of image in a sequence
        vid, key_frame_id = idx[0]
        _, cur_frame_id = idx[1]
        vid_info = self.vid_infos[vid]

        # load image
        key_img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames_all'][key_frame_id]))
        cur_img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames_all'][cur_frame_id]))
        h_orig, w_orig, _ = key_img.shape
        basename = osp.basename(vid_info['filenames_all'][key_frame_id])

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        cur_img, img_shape, pad_shape, scale_factor = self.img_transform(
            cur_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        if (type(scale_factor)) != float:
            scale_factor = tuple(scale_factor)
        cur_img = cur_img.copy()
        key_img, key_img_shape, _, ref_scale_factor = self.img_transform(
            key_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        key_img = key_img.copy()

        # trans = torchvision.transforms.ToTensor()
        key_img = torch.from_numpy(key_img).cuda()
        cur_img = torch.from_numpy(cur_img).cuda()

        def resize(feat_map, size=(48, 64)):
            """Resize feature map to certain size."""
            key_feature = torch.nn.functional.interpolate(feat_map, size, mode='bilinear', align_corners=True)
            return key_feature
        img_size = (384, 640)
        if key_img.shape[-2:] != img_size:
            key_img = resize(key_img.unsqueeze(0), img_size).squeeze(0)
            cur_img = resize(cur_img.unsqueeze(0), img_size).squeeze(0)

        key_feature_maps, _ = self.det_model.extract_feat(key_img.unsqueeze(0))
        cur_feature_maps, _ = self.det_model.extract_feat(cur_img.unsqueeze(0))

        key_feature_maps = [feat_map.squeeze(0) for feat_map in key_feature_maps]
        cur_feature_maps = [feat_map.squeeze(0) for feat_map in cur_feature_maps]

        data = dict(
            key_img=key_img,
            cur_img=cur_img,
            key_img_feats=key_feature_maps,
            cur_img_feats=cur_feature_maps
        )
        return data

    def prepare_train_flow_img(self, idx):

        # prepare a pair of image in a sequence
        vid, key_frame_id = idx[0]
        _, cur_frame_id = idx[1]
        vid_info = self.vid_infos[vid]

        # load image
        key_img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames_all'][key_frame_id]))
        cur_img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames_all'][cur_frame_id]))
        h_orig, w_orig, _ = cur_img.shape
        basename = osp.basename(vid_info['filenames_all'][key_frame_id])

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None
        ann_idx = vid_info['filenames'].index(vid_info['filenames_all'][cur_frame_id])
        ann = self.get_ann_info(vid, ann_idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            cur_img, gt_bboxes, gt_labels = self.extra_aug(cur_img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False

        img_scales = [(1280, 720), (640, 360)]
        # img_scale = random_scale(self.img_scales)  # sample a scale
        cur_img, img_shape, pad_shape, scale_factor = self.img_transform(
            cur_img, img_scales[1], flip, keep_ratio=self.resize_keep_ratio)
        if (type(scale_factor)) != float:
            scale_factor = tuple(scale_factor)
        cur_img = cur_img.copy()
        key_img, key_img_shape, _, key_scale_factor = self.img_transform(
            key_img, img_scales[0], flip, keep_ratio=self.resize_keep_ratio)
        key_img = key_img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)

        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            if w_orig > h_orig:
                h, w = img_shape[0], img_shape[1]
                _scale_factor = tuple([w, h, w, h])
            else:
                _scale_factor = scale_factor
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           _scale_factor, flip)

        ori_shape = (vid_info['height'], vid_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            is_first=(cur_frame_id == 0),
            flip=flip)

        data = dict(
            img=DC(to_tensor(key_img), stack=True),
            ref_img=DC(to_tensor(cur_img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            # ref_bboxes=DC(to_tensor(ref_bboxes))
        )
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        # if self.with_track:
        #     data['gt_pids'] = DC(to_tensor(gt_pids))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        data['train_flow'] = True

        if self.cuda:
            key_img_cuda = torch.from_numpy(key_img).cuda()
            cur_img_cuda = torch.from_numpy(cur_img).cuda()

            def resize(feat_map, size=(48, 64)):
                """Resize feature map to certain size."""
                key_feature = torch.nn.functional.interpolate(feat_map, size, mode='bilinear', align_corners=True)
                return key_feature

            img_size = (384, 640)
            if key_img_cuda.shape[-2:] != img_size:
                key_img_cuda = resize(key_img_cuda.unsqueeze(0), img_size).squeeze(0)
                cur_img_cuda = resize(cur_img_cuda.unsqueeze(0), img_size).squeeze(0)

            key_feature_maps, _ = self.det_model.extract_feat(key_img_cuda.unsqueeze(0))
            cur_feature_maps, _ = self.det_model.extract_feat(cur_img_cuda.unsqueeze(0))

            key_feature_maps = [feat_map.squeeze(0) for feat_map in key_feature_maps]
            cur_feature_maps = [feat_map.squeeze(0) for feat_map in cur_feature_maps]

            data['key_feature_maps'] = key_feature_maps
            data['cur_feature_maps'] = cur_feature_maps

        return data

    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        # load image
        if self.is_flow or self.every_frame:
            img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames_all'][frame_id]))
        else:
            img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
        h_orig, w_orig, _ = img.shape
        basename = osp.basename(vid_info['filenames'][frame_id])
        _, ref_frame_id = self.sample_ref(idx)
        ref_img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][ref_frame_id]))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(vid, frame_id)
        ref_ann = self.get_ann_info(vid, ref_frame_id)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        ref_bboxes = ref_ann['bboxes']
        # obj ids attribute does not exist in current annotation
        # need to add it
        ref_ids = ref_ann['obj_ids']
        gt_ids = ann['obj_ids']
        # compute matching of reference frame with current frame
        # 0 denote there is no matching
        gt_pids = [ref_ids.index(i) + 1 if i in ref_ids else 0 for i in gt_ids]
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        if (type(scale_factor)) != float:
            scale_factor = tuple(scale_factor)
        img = img.copy()
        ref_img, ref_img_shape, _, ref_scale_factor = self.img_transform(
            ref_img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        ref_img = ref_img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        ref_bboxes = self.bbox_transform(ref_bboxes, ref_img_shape, ref_scale_factor,
                                         flip)
        if self.aug_ref_bbox_param is not None:
            ref_bboxes = self.bbox_aug(ref_bboxes, ref_img_shape)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            if w_orig > h_orig:
                h, w = img_shape[0], img_shape[1]
                _scale_factor = tuple([w, h, w, h])
            else:
                _scale_factor = scale_factor
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           _scale_factor, flip)

        ori_shape = (vid_info['height'], vid_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            is_first=(frame_id == 0),
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            ref_img=DC(to_tensor(ref_img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            ref_bboxes=DC(to_tensor(ref_bboxes))
        )
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_track:
            data['gt_pids'] = DC(to_tensor(gt_pids))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        is_anno = True
        if self.every_frame:
            img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames_all'][frame_id]))
            is_anno = vid_info['filenames_all'][frame_id] in vid_info['filenames']
        else:
            img = mmcv.imread(osp.join(self.img_prefix, vid_info['filenames'][frame_id]))
        proposal = None

        if self.every_frame:
            file_name = vid_info['filenames_all'][frame_id]
        else:
            file_name = vid_info['filenames'][frame_id]

        def prepare_single(img, frame_id, scale, flip, file_name, proposal=None, is_anno=True):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(vid_info['height'], vid_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                is_first=(frame_id == 0),
                video_id=vid,
                file_name=file_name,
                frame_id=frame_id,
                scale_factor=scale_factor,
                flip=flip,
                is_anno=is_anno)
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []
        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, frame_id, scale, False, file_name, proposal, is_anno)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)
            if self.flip_ratio > 0:
                _img, _img_meta, _proposal = prepare_single(
                    img, scale, True, file_name, proposal, is_anno)
                imgs.append(_img)
                img_metas.append(DC(_img_meta, cpu_only=True))
                proposals.append(_proposal)
        data = dict(img=imgs, img_meta=img_metas)
        return data

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann
