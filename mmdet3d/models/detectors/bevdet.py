# Copyright (c) Phigent Robotics. All rights reserved.
import os
import sys

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet3d.models.utils import clip_sigmoid
from mmdet.models.backbones.resnet import ResNet


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 use_grid_mask=False,
                 **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = \
                builder.build_backbone(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def _get_expected_bev_hw(self):
        """Get expected BEV feature map size from CenterPoint head configs."""
        cfg = None
        if hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None:
            cfg = getattr(self.pts_bbox_head, 'train_cfg', None) or \
                getattr(self.pts_bbox_head, 'test_cfg', None)
        if cfg is None:
            return None
        grid_size = cfg.get('grid_size', None)
        out_size_factor = cfg.get('out_size_factor', None)
        if grid_size is None or out_size_factor is None:
            return None
        # grid_size is [X, Y, Z], feature map is [H=Y/out, W=X/out].
        target_w = int(grid_size[0] // out_size_factor)
        target_h = int(grid_size[1] // out_size_factor)
        return target_h, target_w

    def _align_bev_feature_size(self, x):
        """Align BEV feature size to head target for stable train/test shapes."""
        expected_hw = self._get_expected_bev_hw()
        if expected_hw is None:
            return x
        target_h, target_w = expected_hw
        cur_h, cur_w = x.shape[-2:]
        if cur_h == target_h and cur_w == target_w:
            return x
        # Keep origin-side alignment; trim extra border cells on max side.
        x = x[..., :min(cur_h, target_h), :min(cur_w, target_w)]
        pad_h = max(0, target_h - x.shape[-2])
        pad_w = max(0, target_w - x.shape[-1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs) # torch.Size([48, 1024, 16, 44])
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        x = self._align_bev_feature_size(x)
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:7])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs


@DETECTORS.register_module()
class BEVDetAux(BEVDet):
    """BEVDet with an auxiliary 2D detection branch on image encoder features.

    The branch projects 3D GT centers onto each camera image and supervises
    a per-camera center head (heatmap + offset regression).
    """

    def __init__(self,
                 aux_2d_head,
                 aux_2d_train_cfg=None,
                 aux_2d_loss_cls=dict(
                     type='GaussianFocalLoss', reduction='mean'),
                 aux_2d_loss_bbox=dict(
                     type='L1Loss', reduction='mean', loss_weight=0.25),
                 aux_2d_loss_weight=1.0,
                 **kwargs):
        super(BEVDetAux, self).__init__(**kwargs)
        self.aux_2d_head = builder.build_head(aux_2d_head)
        self.aux_2d_num_classes = aux_2d_head['heads']['heatmap'][0]
        self.aux_2d_train_cfg = aux_2d_train_cfg or dict(
            max_objs=150,
            min_radius=2,
            min_center_depth=0.5,
            max_center_depth=80.0,
            edge_margin=2.0)
        self.aux_2d_loss_cls = builder.build_loss(aux_2d_loss_cls)
        self.aux_2d_loss_bbox = builder.build_loss(aux_2d_loss_bbox)
        self.aux_2d_loss_weight = aux_2d_loss_weight

    def _extract_feat_with_aux(self, img_inputs):
        """Extract BEV feature and keep 2D encoder features for aux branch."""
        img_inputs = self.prepare_inputs(img_inputs)
        img_feat_2d, _ = self.image_encoder(img_inputs[0])
        x, depth = self.img_view_transformer([img_feat_2d] + img_inputs[1:7])
        x = self.bev_encoder(x)
        return [x], depth, img_feat_2d, img_inputs

    def _gather_feat(self, feat, ind):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        return feat.gather(1, ind)

    def _project_gt_centers_to_multiview(self, gt_bboxes_3d, gt_labels_3d,
                                         img_inputs):
        """Project 3D GT centers to each view and keep valid center targets."""
        imgs, sensor2keyegos, _, intrins, post_rots, post_trans, bda = \
            img_inputs
        bsz, num_cams, _, img_h, img_w = imgs.shape
        min_center_depth = self.aux_2d_train_cfg.get('min_center_depth', 0.5)
        max_center_depth = self.aux_2d_train_cfg.get('max_center_depth', 80.0)
        edge_margin = self.aux_2d_train_cfg.get('edge_margin', 2.0)
        min_box_size = self.aux_2d_train_cfg.get('min_box_size', 2.0)
        device = imgs.device
        centers_2d_list = [[] for _ in range(bsz * num_cams)]
        sizes_2d_list = [[] for _ in range(bsz * num_cams)]
        labels_2d_list = [[] for _ in range(bsz * num_cams)]

        with torch.no_grad():
            for bid in range(bsz):
                labels = gt_labels_3d[bid].to(device).long()
                if labels.numel() == 0:
                    continue

                centers = gt_bboxes_3d[bid].gravity_center.to(
                    device=device, dtype=torch.float32)
                dims = gt_bboxes_3d[bid].dims.to(
                    device=device, dtype=torch.float32)
                yaws = gt_bboxes_3d[bid].yaw.to(
                    device=device, dtype=torch.float32)
                num_obj = centers.shape[0]
                if num_obj == 0:
                    continue

                ones = torch.ones((num_obj, 1), device=device)
                centers_h = torch.cat([centers, ones], dim=-1)
                inv_bda = torch.inverse(bda[bid].to(torch.float32))
                inv_bda_rot = inv_bda[:3, :3]
                # Recover pre-BDA yaw by transforming heading vectors.
                heading_aug = torch.stack(
                    [torch.cos(yaws), torch.sin(yaws), torch.zeros_like(yaws)],
                    dim=1)
                heading_keyego = torch.matmul(heading_aug, inv_bda_rot.t())
                yaw_keyego = torch.atan2(heading_keyego[:, 1],
                                         heading_keyego[:, 0])
                # BEVAug uses isotropic scale. Undo it for image-space size
                # estimation so radius follows physical object size.
                bda_rot = bda[bid, :3, :3].to(torch.float32)
                bda_scale = torch.abs(torch.det(bda_rot)).clamp(
                    min=1e-6).pow(1.0 / 3.0)
                dims_keyego = dims / bda_scale
                centers_keyego = torch.matmul(centers_h, inv_bda.t())[..., :3]
                centers_keyego_h = torch.cat([centers_keyego, ones], dim=-1)

                for cam_id in range(num_cams):
                    keyego2cam = torch.inverse(
                        sensor2keyegos[bid, cam_id].to(torch.float32))
                    centers_cam = torch.matmul(centers_keyego_h,
                                               keyego2cam.t())[..., :3]

                    depths = centers_cam[..., 2]
                    valid = (depths > min_center_depth) & \
                        (depths < max_center_depth)
                    if not valid.any():
                        continue

                    points_img = centers_cam / depths.clamp(
                        min=1e-4).unsqueeze(-1)
                    points_img = torch.matmul(
                        points_img, intrins[bid, cam_id].to(torch.float32).t())
                    points_img = torch.matmul(
                        points_img,
                        post_rots[bid, cam_id].to(torch.float32).t())
                    points_img += post_trans[bid, cam_id].to(
                        torch.float32).view(1, 1, 3)
                    points_img = points_img[..., :2]

                    intrin = intrins[bid, cam_id].to(torch.float32)
                    post_rot = post_rots[bid, cam_id].to(torch.float32)
                    fx = intrin[0, 0]
                    fy = intrin[1, 1]
                    aug_scale_x = torch.norm(post_rot[0, :2], p=2)
                    aug_scale_y = torch.norm(post_rot[1, :2], p=2)
                    cam_rot = keyego2cam[:3, :3]
                    cos_yaw = torch.cos(yaw_keyego)
                    sin_yaw = torch.sin(yaw_keyego)
                    rot_kobj = torch.zeros((num_obj, 3, 3),
                                           dtype=torch.float32,
                                           device=device)
                    rot_kobj[:, 0, 0] = cos_yaw
                    rot_kobj[:, 0, 1] = -sin_yaw
                    rot_kobj[:, 1, 0] = sin_yaw
                    rot_kobj[:, 1, 1] = cos_yaw
                    rot_kobj[:, 2, 2] = 1.0
                    rot_cobj = torch.einsum('ij,njk->nik', cam_rot, rot_kobj)
                    half_dims = 0.5 * dims_keyego
                    # Oriented box half-extent projected to camera axes.
                    ext_cam = torch.matmul(
                        torch.abs(rot_cobj), half_dims.unsqueeze(-1)).squeeze(-1)
                    proj_w = 2.0 * fx * ext_cam[:, 0] / depths.clamp(min=1e-4)
                    proj_h = 2.0 * fy * ext_cam[:, 1] / depths.clamp(min=1e-4)
                    proj_w = proj_w * aug_scale_x
                    proj_h = proj_h * aug_scale_y
                    valid = valid & \
                        (points_img[..., 0] >= edge_margin) & \
                        (points_img[..., 0] <= (img_w - 1 - edge_margin)) & \
                        (points_img[..., 1] >= edge_margin) & \
                        (points_img[..., 1] <= (img_h - 1 - edge_margin)) & \
                        (proj_w >= min_box_size) & \
                        (proj_h >= min_box_size)
                    if not valid.any():
                        continue

                    list_id = bid * num_cams + cam_id
                    valid_ids = valid.nonzero(as_tuple=False).squeeze(1)
                    for obj_id in valid_ids:
                        centers_2d_list[list_id].append(points_img[obj_id])
                        sizes_2d_list[list_id].append(
                            torch.stack([proj_w[obj_id], proj_h[obj_id]]))
                        labels_2d_list[list_id].append(labels[obj_id])

        for idx in range(bsz * num_cams):
            if centers_2d_list[idx]:
                centers_2d_list[idx] = torch.stack(centers_2d_list[idx], dim=0)
                sizes_2d_list[idx] = torch.stack(sizes_2d_list[idx], dim=0)
                labels_2d_list[idx] = torch.stack(labels_2d_list[idx], dim=0)
            else:
                centers_2d_list[idx] = torch.zeros((0, 2), device=device)
                sizes_2d_list[idx] = torch.zeros((0, 2), device=device)
                labels_2d_list[idx] = torch.zeros(
                    (0, ), dtype=torch.long, device=device)
        return centers_2d_list, sizes_2d_list, labels_2d_list

    def _build_aux_2d_targets(self, centers_2d_list, sizes_2d_list,
                              labels_2d_list, feat_h, feat_w, img_h, img_w,
                              device):
        max_objs = self.aux_2d_train_cfg.get('max_objs', 150)
        min_radius = max(int(self.aux_2d_train_cfg.get('min_radius', 2)), 1)
        min_overlap = self.aux_2d_train_cfg.get('min_overlap', 0.3)
        num_imgs = len(centers_2d_list)

        heatmaps = torch.zeros((num_imgs, self.aux_2d_num_classes, feat_h,
                                feat_w),
                               dtype=torch.float32,
                               device=device)
        reg_targets = torch.zeros((num_imgs, max_objs, 2),
                                  dtype=torch.float32,
                                  device=device)
        inds = torch.zeros((num_imgs, max_objs), dtype=torch.long, device=device)
        masks = torch.zeros((num_imgs, max_objs),
                            dtype=torch.bool,
                            device=device)

        scale_x = float(feat_w) / float(img_w)
        scale_y = float(feat_h) / float(img_h)

        for img_id, (centers, sizes, labels) in enumerate(
                zip(centers_2d_list, sizes_2d_list, labels_2d_list)):
            if centers.numel() == 0:
                continue
            scale = centers.new_tensor([scale_x, scale_y])
            valid_obj = 0
            num_objs = min(centers.shape[0], max_objs)
            for obj_id in range(num_objs):
                cls_id = int(labels[obj_id].item())
                if cls_id < 0 or cls_id >= self.aux_2d_num_classes:
                    continue

                center = centers[obj_id] * scale
                center_int = center.to(torch.int32)
                if not (0 <= center_int[0] < feat_w and
                        0 <= center_int[1] < feat_h):
                    continue

                width = sizes[obj_id, 0] * scale_x
                height = sizes[obj_id, 1] * scale_y
                radius = gaussian_radius((height, width), min_overlap)
                radius = max(min_radius, int(radius))
                draw_heatmap_gaussian(heatmaps[img_id, cls_id], center_int,
                                      radius)
                x_int, y_int = center_int[0], center_int[1]
                inds[img_id, valid_obj] = y_int * feat_w + x_int
                masks[img_id, valid_obj] = 1
                reg_targets[img_id, valid_obj, 0] = \
                    center[0] - x_int.to(torch.float32)
                reg_targets[img_id, valid_obj, 1] = \
                    center[1] - y_int.to(torch.float32)
                valid_obj += 1
                if valid_obj >= max_objs:
                    break
        return heatmaps, reg_targets, inds, masks

    def forward_img_auxiliary_train(self, img_feat_2d, gt_bboxes_3d,
                                    gt_labels_3d, img_inputs):
        bsz, num_cams, channels, feat_h, feat_w = img_feat_2d.shape
        img_h, img_w = img_inputs[0].shape[-2:]
        feats = img_feat_2d.reshape(bsz * num_cams, channels, feat_h, feat_w)
        preds = self.aux_2d_head(feats)

        centers_2d_list, sizes_2d_list, labels_2d_list = \
            self._project_gt_centers_to_multiview(
            gt_bboxes_3d, gt_labels_3d, img_inputs)
        heatmaps, reg_targets, inds, masks = self._build_aux_2d_targets(
            centers_2d_list, sizes_2d_list, labels_2d_list, feat_h, feat_w,
            img_h, img_w, feats.device)

        heatmap_pred = clip_sigmoid(preds['heatmap'])
        num_pos = heatmaps.eq(1).float().sum().item()
        cls_avg_factor = max(num_pos, 1.0)
        loss_heatmap = self.aux_2d_loss_cls(
            heatmap_pred, heatmaps, avg_factor=cls_avg_factor)

        offset_pred = preds['offset'].permute(0, 2, 3, 1).contiguous()
        offset_pred = offset_pred.reshape(
            offset_pred.size(0), -1, offset_pred.size(3))
        offset_pred = self._gather_feat(offset_pred, inds)
        offset_targets = reg_targets
        reg_mask = masks.unsqueeze(2).float()
        num_reg = max(reg_mask.sum().item(), 1.0)
        loss_offset = self.aux_2d_loss_bbox(
            offset_pred,
            offset_targets,
            reg_mask.expand_as(offset_targets),
            avg_factor=num_reg)

        loss_weight = self.aux_2d_loss_weight
        return dict(
            loss_aux2d_heatmap=loss_heatmap * loss_weight,
            loss_aux2d_offset=loss_offset * loss_weight)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        img_feats, _, img_feat_2d, prepared_inputs = \
            self._extract_feat_with_aux(img_inputs)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        losses_aux2d = self.forward_img_auxiliary_train(
            img_feat_2d, gt_bboxes_3d, gt_labels_3d, prepared_inputs)
        losses.update(losses_aux2d)
        return losses


@DETECTORS.register_module()
class BEVDetTRT(BEVDet):

    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        if getattr(self, 'force_depth_fp32', False):
            x_fp32 = x.float()
            x = self.img_view_transformer.depth_net(x_fp32).to(x.dtype)
        else:
            x = self.img_view_transformer.depth_net(x)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        input = self.prepare_inputs(input)
        coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class BEVDet4D(BEVDet):
    r"""BEVDet4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 **kwargs):
        super(BEVDet4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.grid = None

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        bda_rot = bda[..., :3, :3] if bda.shape[-1] == 4 else bda
        bda_[:, :, :3, :3] = bda_rot.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_adj_rot = \
                bda_adj[..., :3, :3] if bda_adj.shape[-1] == 4 else bda_adj
            bda_[:, :, :3, :3] = bda_adj_rot.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1,
                                                            2) * 2.0 - 1.0
        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input):
        x, _ = self.image_encoder(img) # torch.Size([8, 6, 512, 16, 44])
        bev_feat, depth = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]
        bev_feat_list = []
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...],
            intrins, post_rots, post_trans, bda[0:1, ...])
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # align the feat_prev
        _, C, H, W = feat_prev.shape
        feat_prev = \
            self.shift_feature(feat_prev,
                               [sensor2keyegos_curr, sensor2keyegos_prev],
                               bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, _ = self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        depth_list = []
        key_frame = True  # back propagation for key frame only
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot,
                               post_tran, bda, mlp_input)
                if key_frame:
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                else:
                    with torch.no_grad():
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_list[0]


@DETECTORS.register_module()
class BEVDepth4D(BEVDet4D):

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVDepth4DDepthKD(BEVDepth4D):
    """BEVDepth4D with additional depth distillation loss."""

    def __init__(self, depth_kd_cfg=None, **kwargs):
        super(BEVDepth4DDepthKD, self).__init__(**kwargs)
        depth_kd_cfg = depth_kd_cfg or dict()
        self.depth_kd_enabled = depth_kd_cfg.get('enabled', True)
        self.depth_kd_loss_weight = float(depth_kd_cfg.get('loss_weight', 1.0))
        self.depth_kd_loss_type = depth_kd_cfg.get('loss_type', 'kl')
        self.depth_kd_temperature = float(depth_kd_cfg.get('temperature', 1.0))
        self.depth_kd_teacher_source = depth_kd_cfg.get('teacher_source',
                                                        'tensor')
        self.depth_kd_teacher_key = depth_kd_cfg.get('teacher_key', 'gt_depth')
        self.depth_kd_teacher_is_logits = bool(
            depth_kd_cfg.get('teacher_is_logits', False))
        self.depth_kd_teacher_depth_mode = depth_kd_cfg.get(
            'teacher_depth_mode', 'metric')
        self.depth_kd_ignore_if_missing = bool(
            depth_kd_cfg.get('ignore_if_missing', True))
        self.depth_kd_use_fg_mask = bool(
            depth_kd_cfg.get('use_fg_mask', True))
        self.depth_kd_grad_loss_weight = float(
            depth_kd_cfg.get('grad_loss_weight', 0.0))
        self.depth_kd_relative_loss_weight = float(
            depth_kd_cfg.get('relative_loss_weight', 0.0))
        self.depth_kd_relative_temperature = float(
            depth_kd_cfg.get('relative_temperature', 1.0))
        self.depth_kd_relative_min_diff = float(
            depth_kd_cfg.get('relative_min_diff', 0.0))
        self.depth_kd_relative_use_log_depth = bool(
            depth_kd_cfg.get('relative_use_log_depth', False))
        self.depth_kd_ray_loss_weight = float(
            depth_kd_cfg.get('ray_loss_weight', 0.0))
        self.depth_kd_ray_loss_type = depth_kd_cfg.get('ray_loss_type', 'l1')
        # Dense spatial KD: keep original LiDAR depth loss path untouched,
        # but compute KD map losses on high-resolution teacher maps.
        self.depth_kd_dense_spatial = bool(
            depth_kd_cfg.get('dense_spatial', False))
        self.depth_kd_dense_base_loss = bool(
            depth_kd_cfg.get('dense_base_loss', False))
        self.depth_kd_dense_interp = depth_kd_cfg.get('dense_interp',
                                                      'bilinear')
        if self.depth_kd_loss_type not in ['kl', 'l1']:
            raise ValueError('depth_kd_cfg.loss_type must be "kl" or "l1"')
        if self.depth_kd_teacher_depth_mode not in ['metric', 'relative']:
            raise ValueError(
                'depth_kd_cfg.teacher_depth_mode must be "metric" '
                'or "relative"')
        if self.depth_kd_ray_loss_type not in ['l1', 'smooth_l1']:
            raise ValueError(
                'depth_kd_cfg.ray_loss_type must be "l1" or "smooth_l1"')
        if self.depth_kd_dense_interp not in ['bilinear', 'nearest']:
            raise ValueError(
                'depth_kd_cfg.dense_interp must be "bilinear" or "nearest"')
        if self.depth_kd_teacher_source not in ['tensor',
                                                'depth_anything_v3']:
            raise ValueError(
                'depth_kd_cfg.teacher_source must be "tensor" or '
                '"depth_anything_v3"')

        da3_cfg = depth_kd_cfg.get('depth_anything_v3', dict())
        self.da3_repo_path = da3_cfg.get('repo_path', 'Depth-Anything-3/src')
        self.da3_model_name = da3_cfg.get('model_name', 'da3mono-large')
        self.da3_pretrained = da3_cfg.get('pretrained', None)
        self.da3_allow_random_init = bool(
            da3_cfg.get('allow_random_init', False))
        self.da3_device = da3_cfg.get('device', 'same')
        self.da3_patch_size = int(da3_cfg.get('patch_size', 14))
        self.da3_color_order = da3_cfg.get('color_order', 'rgb')
        self.da3_input_size = da3_cfg.get('input_size', None)
        self.da3_align_to_gt = bool(da3_cfg.get('align_to_gt', True))
        self.da3_min_align_pixels = int(da3_cfg.get('min_align_pixels', 32))
        self.da3_ref_view_strategy = da3_cfg.get('ref_view_strategy',
                                                 'saddle_balanced')
        if self.da3_color_order not in ['rgb', 'bgr']:
            raise ValueError(
                'depth_kd_cfg.depth_anything_v3.color_order must be "rgb" '
                'or "bgr"')
        if self.da3_input_size is not None:
            if (not isinstance(self.da3_input_size, (list, tuple)) or
                    len(self.da3_input_size) != 2):
                raise ValueError(
                    'depth_kd_cfg.depth_anything_v3.input_size must be '
                    'None or [H, W]')
            self.da3_input_size = (
                int(self.da3_input_size[0]), int(self.da3_input_size[1]))

        self._da3_teacher = None
        self._da3_teacher_device = None

    def _project_root(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

    def _resolve_da3_repo_path(self):
        repo_path = self.da3_repo_path
        if repo_path is None:
            return None
        if not os.path.isabs(repo_path):
            repo_path = os.path.join(self._project_root(), repo_path)
        return os.path.abspath(repo_path)

    def _resolve_da3_device(self, student_device):
        if self.da3_device == 'same':
            return student_device
        device = torch.device(self.da3_device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            return student_device
        return device

    def _load_depth_anything_teacher(self, device):
        repo_path = self._resolve_da3_repo_path()
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                'Failed to import Depth-Anything-3. Set '
                'depth_kd_cfg.depth_anything_v3.repo_path correctly and '
                'install dependencies (e.g. `pip install -r '
                'Depth-Anything-3/requirements.txt`).'
            ) from exc

        if self.da3_pretrained:
            teacher = DepthAnything3.from_pretrained(self.da3_pretrained)
        else:
            if not self.da3_allow_random_init:
                raise ValueError(
                    'Depth-Anything-3 pretrained checkpoint is required. '
                    'Set depth_kd_cfg.depth_anything_v3.pretrained or '
                    'set allow_random_init=True.')
            teacher = DepthAnything3(model_name=self.da3_model_name)

        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _get_depth_anything_teacher(self, student_device):
        teacher_device = self._resolve_da3_device(student_device)
        if self._da3_teacher is None:
            self._da3_teacher = self._load_depth_anything_teacher(
                teacher_device)
            self._da3_teacher_device = teacher_device
            return self._da3_teacher, teacher_device
        if self._da3_teacher_device != teacher_device:
            self._da3_teacher = self._da3_teacher.to(teacher_device)
            self._da3_teacher_device = teacher_device
        return self._da3_teacher, teacher_device

    def _get_keyframe_images(self, img_inputs):
        imgs = img_inputs[0]
        B, N_all, C, H, W = imgs.shape
        if N_all % self.num_frame != 0:
            raise ValueError(
                f'Invalid image count {N_all} for num_frame={self.num_frame}')
        num_cams = N_all // self.num_frame
        return imgs[:, :num_cams], (H, W)

    def _prepare_da3_images(self, key_imgs):
        da3_imgs = key_imgs
        if self.da3_color_order == 'bgr':
            da3_imgs = da3_imgs[:, :, [2, 1, 0], :, :]

        B, N, C, H, W = da3_imgs.shape
        da3_imgs = da3_imgs.view(B * N, C, H, W)
        if self.da3_input_size is not None:
            da3_imgs = F.interpolate(
                da3_imgs,
                size=self.da3_input_size,
                mode='bilinear',
                align_corners=False)
        H, W = da3_imgs.shape[-2:]
        if self.da3_patch_size > 1:
            crop_h = (H // self.da3_patch_size) * self.da3_patch_size
            crop_w = (W // self.da3_patch_size) * self.da3_patch_size
            if crop_h <= 0 or crop_w <= 0:
                raise ValueError(
                    f'Invalid DA3 crop size ({crop_h}, {crop_w}) from '
                    f'input ({H}, {W}) and patch_size={self.da3_patch_size}')
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2
            da3_imgs = da3_imgs[:, :, top:top + crop_h, left:left + crop_w]
        H, W = da3_imgs.shape[-2:]
        return da3_imgs.view(B, N, C, H, W)

    def _align_teacher_depth_to_gt(self, teacher_depth, gt_depth):
        if gt_depth is None:
            return teacher_depth
        aligned_depth = teacher_depth.clone()
        B, N = teacher_depth.shape[:2]
        eps = 1e-6
        for b in range(B):
            for n in range(N):
                t = teacher_depth[b, n]
                g = gt_depth[b, n].to(t.device)
                valid = (t > 0.0) & (g > 0.0) & torch.isfinite(t) & \
                    torch.isfinite(g)
                if valid.sum() < self.da3_min_align_pixels:
                    continue
                t_med = torch.median(t[valid])
                g_med = torch.median(g[valid])
                if t_med > eps and torch.isfinite(g_med):
                    aligned_depth[b, n] = t * (g_med / t_med)
        return aligned_depth

    @torch.no_grad()
    def _infer_teacher_depth_from_da3(self, img_inputs, gt_depth=None):
        key_imgs, (H, W) = self._get_keyframe_images(img_inputs)
        teacher, teacher_device = self._get_depth_anything_teacher(
            key_imgs.device)
        da3_imgs = self._prepare_da3_images(key_imgs).to(
            teacher_device, non_blocking=True).float()
        raw_output = teacher(
            da3_imgs,
            ref_view_strategy=self.da3_ref_view_strategy,
        )
        teacher_depth = raw_output.get('depth', None)
        if teacher_depth is None:
            raise RuntimeError('Depth-Anything-3 output missing "depth".')
        if teacher_depth.dim() == 5 and teacher_depth.shape[2] == 1:
            teacher_depth = teacher_depth.squeeze(2)
        if teacher_depth.dim() != 4:
            raise RuntimeError(
                f'Unexpected DA3 depth shape: {tuple(teacher_depth.shape)}')
        teacher_depth = teacher_depth.to(key_imgs.device).float()
        dH, dW = teacher_depth.shape[-2:]
        if (dH, dW) != (H, W):
            B, N = teacher_depth.shape[:2]
            teacher_depth = F.interpolate(
                teacher_depth.view(B * N, 1, dH, dW),
                size=(H, W),
                mode='bilinear',
                align_corners=False).view(B, N, H, W)
        teacher_depth = torch.where(
            torch.isfinite(teacher_depth), teacher_depth,
            torch.zeros_like(teacher_depth))
        teacher_depth = teacher_depth.clamp(min=0.0)
        if self.da3_align_to_gt:
            teacher_depth = self._align_teacher_depth_to_gt(
                teacher_depth, gt_depth)
        return teacher_depth

    def _metric_depth_to_prob(self, depth_map, target_img_hw=None):
        """Convert metric depth map [B, N, H, W] to prob volume [BN, D, h, w]."""
        B, N, H, W = depth_map.shape
        if target_img_hw is not None and (H, W) != tuple(target_img_hw):
            tgt_h, tgt_w = int(target_img_hw[0]), int(target_img_hw[1])
            depth_map = F.interpolate(
                depth_map.view(B * N, 1, H, W),
                size=(tgt_h, tgt_w),
                mode='nearest').view(B, N, tgt_h, tgt_w)
            H, W = tgt_h, tgt_w
        h = H // self.img_view_transformer.downsample
        w = W // self.img_view_transformer.downsample
        depth_prob = self.img_view_transformer.get_downsampled_gt_depth(
            depth_map)
        depth_prob = depth_prob.view(B * N, h, w, self.img_view_transformer.D)
        return depth_prob.permute(0, 3, 1, 2).contiguous()

    def _prepare_teacher_depth(self, teacher_depth, depth_preds):
        """Prepare teacher depth as [BN, D, h, w] probabilities."""
        if isinstance(teacher_depth, (list, tuple)):
            if len(teacher_depth) == 1:
                teacher_depth = teacher_depth[0]
            else:
                raise ValueError('teacher depth list/tuple must have length 1')

        if not torch.is_tensor(teacher_depth):
            teacher_depth = torch.as_tensor(teacher_depth)
        teacher_depth = teacher_depth.to(depth_preds.device)
        target_img_hw = (
            depth_preds.shape[2] * self.img_view_transformer.downsample,
            depth_preds.shape[3] * self.img_view_transformer.downsample)

        if teacher_depth.dim() == 4:
            if teacher_depth.shape[1] == self.img_view_transformer.D and \
                    teacher_depth.shape[2:] == depth_preds.shape[2:]:
                teacher_prob = teacher_depth
            else:
                teacher_prob = self._metric_depth_to_prob(
                    teacher_depth, target_img_hw=target_img_hw)
        elif teacher_depth.dim() == 5 and \
                teacher_depth.shape[2] == self.img_view_transformer.D and \
                teacher_depth.shape[3:] == depth_preds.shape[2:]:
            teacher_prob = teacher_depth.flatten(0, 1)
        else:
            raise ValueError(
                'Unsupported teacher depth shape. Expected [B,N,H,W], '
                '[BN,D,h,w], or [B,N,D,h,w].')
        if teacher_prob.shape[2:] != depth_preds.shape[2:]:
            teacher_prob = F.interpolate(
                teacher_prob,
                size=depth_preds.shape[2:],
                mode='bilinear',
                align_corners=False)

        eps = 1e-6
        if self.depth_kd_teacher_is_logits:
            teacher_prob = F.softmax(
                teacher_prob / max(self.depth_kd_temperature, eps), dim=1)
        else:
            teacher_prob = teacher_prob.clamp(min=0.0)
            teacher_prob = teacher_prob / teacher_prob.sum(
                dim=1, keepdim=True).clamp(min=eps)
        return teacher_prob

    def _prepare_teacher_depth_map(self,
                                   teacher_depth,
                                   depth_preds,
                                   target_hw=None):
        """Prepare teacher depth map as [BN, h, w] (no depth-bin conversion)."""
        if isinstance(teacher_depth, (list, tuple)):
            if len(teacher_depth) == 1:
                teacher_depth = teacher_depth[0]
            else:
                raise ValueError('teacher depth list/tuple must have length 1')

        if not torch.is_tensor(teacher_depth):
            teacher_depth = torch.as_tensor(teacher_depth)
        teacher_depth = teacher_depth.to(depth_preds.device).float()

        if teacher_depth.dim() == 4 and \
                teacher_depth.shape[1] == self.img_view_transformer.D and \
                teacher_depth.shape[2:] == depth_preds.shape[2:]:
            # [BN, D, h, w] depth probability/logit volume.
            teacher_map = self._prob_to_expected_depth(teacher_depth)
        elif teacher_depth.dim() == 5 and \
                teacher_depth.shape[2] == self.img_view_transformer.D and \
                teacher_depth.shape[3:] == depth_preds.shape[2:]:
            # [B, N, D, h, w] depth probability/logit volume.
            teacher_map = self._prob_to_expected_depth(
                teacher_depth.flatten(0, 1))
        elif teacher_depth.dim() == 4:
            B, N = teacher_depth.shape[:2]
            teacher_map = teacher_depth.view(B * N, teacher_depth.shape[2],
                                             teacher_depth.shape[3])
        elif teacher_depth.dim() == 3:
            teacher_map = teacher_depth
        else:
            raise ValueError(
                'Unsupported teacher depth shape for map mode. '
                'Expected [B,N,H,W] or [BN,H,W].')

        if target_hw is None:
            target_hw = teacher_map.shape[1:]
        if teacher_map.shape[1:] != tuple(target_hw):
            teacher_map = F.interpolate(
                teacher_map.unsqueeze(1),
                size=tuple(target_hw),
                mode='bilinear',
                align_corners=False).squeeze(1)
        teacher_map = torch.where(
            torch.isfinite(teacher_map), teacher_map,
            torch.zeros_like(teacher_map))
        return teacher_map.clamp(min=0.0)

    def _get_depth_bin_values(self, device, dtype):
        depth_cfg = self.img_view_transformer.grid_config['depth']
        d_min, d_max, d_step = float(depth_cfg[0]), float(depth_cfg[1]), \
            float(depth_cfg[2])
        D = int(self.img_view_transformer.D)
        if self.img_view_transformer.sid:
            if D <= 1:
                vals = torch.tensor([d_min], device=device, dtype=dtype)
            else:
                idx = torch.arange(D, device=device, dtype=dtype)
                vals = torch.exp(
                    torch.log(torch.tensor(d_min, device=device, dtype=dtype))
                    + idx / (D - 1) * torch.log(
                        torch.tensor((d_max - 1.0) / d_min,
                                     device=device,
                                     dtype=dtype)))
        else:
            vals = d_min + torch.arange(
                D, device=device, dtype=dtype) * d_step
        return vals

    def _prob_to_expected_depth(self, prob):
        # prob: [BN, D, h, w] -> expected depth: [BN, h, w]
        bin_vals = self._get_depth_bin_values(prob.device, prob.dtype)
        return (prob * bin_vals.view(1, -1, 1, 1)).sum(dim=1)

    def _build_depth_fg_mask(self, teacher_prob, gt_depth=None):
        if self.depth_kd_use_fg_mask and gt_depth is not None:
            fg = self.img_view_transformer.get_downsampled_gt_depth(gt_depth)
            fg = (fg.max(dim=1).values > 0.0).to(teacher_prob.dtype)
            fg = fg.view(teacher_prob.shape[0], teacher_prob.shape[2],
                         teacher_prob.shape[3])
            return fg
        return (teacher_prob.sum(dim=1) > 0.0).to(teacher_prob.dtype)

    def _build_depth_fg_mask_from_map(self, teacher_map, gt_depth=None):
        if self.depth_kd_use_fg_mask and gt_depth is not None:
            fg = self.img_view_transformer.get_downsampled_gt_depth(gt_depth)
            fg = (fg.max(dim=1).values > 0.0).to(teacher_map.dtype)
            fg = fg.view(teacher_map.shape[0], teacher_map.shape[1],
                         teacher_map.shape[2])
            return fg
        return (teacher_map > 0.0).to(teacher_map.dtype)

    def _normalize_depth_map_by_fg_median(self, depth_map, fg_mask):
        """Scale each map by its foreground median for relative-depth losses."""
        out = depth_map.clone()
        eps = 1e-6
        for i in range(out.shape[0]):
            d = out[i]
            m = fg_mask[i] > 0.0
            valid = m & torch.isfinite(d) & (d > 0.0)
            if valid.sum() == 0:
                continue
            med = torch.median(d[valid])
            if torch.isfinite(med) and med > eps:
                out[i] = d / med
        return out

    def _distill_gradient_loss(self, student_depth, teacher_depth, fg_mask):
        # depth: [BN, h, w], fg_mask: [BN, h, w]
        eps = 1e-6
        dx_s = student_depth[:, :, 1:] - student_depth[:, :, :-1]
        dx_t = teacher_depth[:, :, 1:] - teacher_depth[:, :, :-1]
        mx = fg_mask[:, :, 1:] * fg_mask[:, :, :-1]
        dy_s = student_depth[:, 1:, :] - student_depth[:, :-1, :]
        dy_t = teacher_depth[:, 1:, :] - teacher_depth[:, :-1, :]
        my = fg_mask[:, 1:, :] * fg_mask[:, :-1, :]

        loss_x = (dx_s - dx_t).abs() * mx
        loss_y = (dy_s - dy_t).abs() * my
        denom = (mx.sum() + my.sum()).clamp(min=eps)
        return (loss_x.sum() + loss_y.sum()) / denom

    def _distill_relative_loss(self, student_depth, teacher_depth, fg_mask):
        # Local ordinal KD on 4-neighbor pairs (right/down).
        eps = 1e-6
        tau = max(self.depth_kd_relative_temperature, eps)
        min_diff = self.depth_kd_relative_min_diff

        if self.depth_kd_relative_use_log_depth:
            student_depth = torch.log(student_depth.clamp(min=eps))
            teacher_depth = torch.log(teacher_depth.clamp(min=eps))

        def _dir_loss(sd, td, m):
            valid = (m > 0.0) & torch.isfinite(sd) & torch.isfinite(td) & \
                (td.abs() > min_diff)
            if valid.sum() == 0:
                return sd.new_tensor(0.0), sd.new_tensor(0.0)
            sign_t = torch.sign(td[valid])
            logits = sign_t * sd[valid] / tau
            # logistic ranking loss
            loss = F.softplus(-logits).sum()
            denom = valid.sum().to(sd.dtype)
            return loss, denom

        dx_s = student_depth[:, :, 1:] - student_depth[:, :, :-1]
        dx_t = teacher_depth[:, :, 1:] - teacher_depth[:, :, :-1]
        mx = fg_mask[:, :, 1:] * fg_mask[:, :, :-1]
        loss_x, den_x = _dir_loss(dx_s, dx_t, mx)

        dy_s = student_depth[:, 1:, :] - student_depth[:, :-1, :]
        dy_t = teacher_depth[:, 1:, :] - teacher_depth[:, :-1, :]
        my = fg_mask[:, 1:, :] * fg_mask[:, :-1, :]
        loss_y, den_y = _dir_loss(dy_s, dy_t, my)

        denom = (den_x + den_y).clamp(min=eps)
        return (loss_x + loss_y) / denom

    def _distill_ray_loss(self, student_prob, teacher_prob, fg_mask):
        # Distill cumulative transmittance along depth bins (camera-ray KD).
        eps = 1e-6
        s_trans = 1.0 - torch.cumsum(student_prob, dim=1)
        t_trans = 1.0 - torch.cumsum(teacher_prob, dim=1)
        mask = fg_mask.unsqueeze(1).expand_as(s_trans)
        if self.depth_kd_ray_loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(
                s_trans, t_trans, reduction='none', beta=1.0)
        else:
            loss = (s_trans - t_trans).abs()
        denom = mask.sum().clamp(min=eps)
        return (loss * mask).sum() / denom

    def _get_depth_kd_core(self, depth_preds, teacher_depth, gt_depth=None):
        teacher_prob = self._prepare_teacher_depth(teacher_depth, depth_preds)
        eps = 1e-6

        student_prob = depth_preds.clamp(min=eps, max=1.0)
        if self.depth_kd_temperature != 1.0:
            inv_temp = 1.0 / max(self.depth_kd_temperature, eps)
            student_prob = student_prob.pow(inv_temp)
            student_prob = student_prob / student_prob.sum(
                dim=1, keepdim=True).clamp(min=eps)

        fg = self._build_depth_fg_mask(teacher_prob, gt_depth=gt_depth)
        return student_prob, teacher_prob, fg

    @force_fp32()
    def get_depth_kd_loss(self, depth_preds, teacher_depth, gt_depth=None):
        kd_losses = self.get_depth_kd_losses(
            depth_preds=depth_preds,
            teacher_depth=teacher_depth,
            gt_depth=gt_depth)
        return kd_losses['loss_depth_kd']

    @force_fp32()
    def get_depth_kd_losses(self, depth_preds, teacher_depth, gt_depth=None):
        # 1) Top-level branch by teacher mode.
        mode = self.depth_kd_teacher_depth_mode  # 'metric' or 'relative'

        need_prob = self.depth_kd_ray_loss_weight > 0.0
        need_map = (self.depth_kd_loss_weight > 0.0) or \
            (self.depth_kd_grad_loss_weight > 0.0) or \
            (self.depth_kd_relative_loss_weight > 0.0)

        student_prob, teacher_prob, fg_prob = None, None, None
        if need_prob:
            student_prob, teacher_prob, fg_prob = self._get_depth_kd_core(
                depth_preds=depth_preds,
                teacher_depth=teacher_depth,
                gt_depth=gt_depth)
        else:
            eps = 1e-6
            student_prob = depth_preds.clamp(min=eps, max=1.0)
            student_prob = student_prob / student_prob.sum(
                dim=1, keepdim=True).clamp(min=eps)

        student_depth_map, teacher_depth_map, fg_map = None, None, None
        if need_map:
            student_depth_map = self._prob_to_expected_depth(student_prob)
            # Keep student map resolution and downsample teacher map to match.
            teacher_depth_map = self._prepare_teacher_depth_map(
                teacher_depth,
                depth_preds,
                target_hw=student_depth_map.shape[1:])
            fg_map = self._build_depth_fg_mask_from_map(
                teacher_depth_map, gt_depth=gt_depth)
            if mode == 'relative':
                # Relative teacher has arbitrary per-view scale.
                student_depth_map = self._normalize_depth_map_by_fg_median(
                    student_depth_map, fg_map)
                teacher_depth_map = self._normalize_depth_map_by_fg_median(
                    teacher_depth_map, fg_map)

        losses = dict(loss_depth_kd=depth_preds.new_tensor(0.0))
        if self.depth_kd_loss_weight > 0.0:
            loss = (student_depth_map - teacher_depth_map).abs()
            denom = fg_map.sum().clamp(min=1.0)
            loss_base = (loss * fg_map).sum() / denom
            losses['loss_depth_kd'] = loss_base * self.depth_kd_loss_weight

        # Ray KD is independent from base representation branch.
        if self.depth_kd_ray_loss_weight > 0.0:
            loss_ray = self._distill_ray_loss(
                student_prob, teacher_prob, fg_prob)
            losses['loss_depth_kd_ray'] = \
                loss_ray * self.depth_kd_ray_loss_weight

        # Gradient / local-relative KD are independent optional heads.
        if self.depth_kd_grad_loss_weight > 0.0:
            loss_grad = self._distill_gradient_loss(
                student_depth_map, teacher_depth_map, fg_map)
            losses['loss_depth_kd_grad'] = \
                loss_grad * self.depth_kd_grad_loss_weight

        if self.depth_kd_relative_loss_weight > 0.0:
            loss_relative = self._distill_relative_loss(
                student_depth_map, teacher_depth_map, fg_map)
            losses['loss_depth_kd_relative'] = \
                loss_relative * self.depth_kd_relative_loss_weight
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)

        if self.depth_kd_enabled:
            if self.depth_kd_teacher_source == 'depth_anything_v3':
                try:
                    teacher_depth = self._infer_teacher_depth_from_da3(
                        img_inputs, gt_depth=gt_depth)
                except Exception:
                    if self.depth_kd_ignore_if_missing:
                        teacher_depth = None
                    else:
                        raise
            else:
                teacher_depth = kwargs.get(self.depth_kd_teacher_key, None)
            if teacher_depth is not None:
                losses_kd = self.get_depth_kd_losses(
                    depth_preds=depth,
                    teacher_depth=teacher_depth,
                    gt_depth=gt_depth)
                losses.update(losses_kd)
            elif not self.depth_kd_ignore_if_missing:
                raise KeyError(
                    f'Missing depth KD teacher key: {self.depth_kd_teacher_key}')

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVDepthDepthKD(BEVDepth4DDepthKD):
    """Single-frame BEVDepth with depth distillation."""

    def __init__(self, num_adj=0, with_prev=False, **kwargs):
        super(BEVDepthDepthKD, self).__init__(
            num_adj=num_adj, with_prev=with_prev, **kwargs)


@DETECTORS.register_module()
class BEVDepth4DTRT(BEVDepth4D):
    """TRT export wrapper for BEVDepth4D.

    This wrapper uses single-frame TRT BEV pool path for export/inference.
    """
    def result_serialize(self, outs):
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def result_deserialize(self, outs):
        outs_ = []
        keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
        for head_id in range(len(outs) // 6):
            outs_head = [dict()]
            for kid, key in enumerate(keys):
                outs_head[0][key] = outs[head_id * 6 + kid]
            outs_.append(outs_head)
        return outs_

    def forward(
        self,
        img,
        mlp_input,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        interval_starts,
        interval_lengths,
    ):
        x = self.img_backbone(img)
        x = self.img_neck(x)
        if getattr(self, 'force_depth_fp32', False):
            x_fp32 = x.float()
            x = self.img_view_transformer.depth_net(
                x_fp32, mlp_input.float()).to(x.dtype)
        else:
            x = self.img_view_transformer.depth_net(x, mlp_input)
        depth = x[:, :self.img_view_transformer.D].softmax(dim=1)
        tran_feat = x[:, self.img_view_transformer.D:(
            self.img_view_transformer.D +
            self.img_view_transformer.out_channels)]
        tran_feat = tran_feat.permute(0, 2, 3, 1)
        x = TRTBEVPoolv2.apply(depth.contiguous(), tran_feat.contiguous(),
                               ranks_depth, ranks_feat, ranks_bev,
                               interval_starts, interval_lengths)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.pre_process:
            x = self.pre_process_net(x)[0]
        if self.num_frame > 1:
            pad = torch.zeros_like(x)
            x = torch.cat([x] + [pad] * (self.num_frame - 1), dim=1)
        bev_feat = self.bev_encoder(x)
        outs = self.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs

    def get_bev_pool_input(self, input):
        # Use key frame inputs to build pooling indices.
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, _ = self.prepare_inputs(input)
        sensor2keyego = sensor2keyegos[0]
        ego2global = ego2globals[0]
        intrin = intrins[0]
        post_rot = post_rots[0]
        post_tran = post_trans[0]
        coor = self.img_view_transformer.get_lidar_coor(
            sensor2keyego, ego2global, intrin, post_rot, post_tran, bda)
        return self.img_view_transformer.voxel_pooling_prepare_v2(coor)


@DETECTORS.register_module()
class BEVStereo4D(BEVDepth4D):
    def __init__(self, **kwargs):
        super(BEVStereo4D, self).__init__(**kwargs)
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame-1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) ==4:
                b,c,h,w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1),
                                  h, w]).to(bev_feat_key), bev_feat_key]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame -
                                       self.extra_ref_frames - 1), z,
                                  h, w]).to(bev_feat_key), bev_feat_key]
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame-2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame-2-adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame
