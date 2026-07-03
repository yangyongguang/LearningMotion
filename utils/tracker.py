"""
    write by sunjing at 2021/11/21
"""
import os
import sys

import numpy as np
from numba import jit
import copy

from collections import deque
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import configs
from utils import matching
from utils.tracking_utils.kalman_filter import KalmanFilter
from utils.tracking_utils.kalman_filter_lstm import KalmanFilterLSTM
from .basetrack import BaseTrack, TrackState
import torch.nn as nn
import ops.iou3d_nums.iou3d_nus_utils as iou3d_nms_utils
from utils.matching import linear_assignment, greedy_assignment


class STracker(object):
    # shared_kalman is put in static global var, all Track using the same shared_kalman or shared_kalman_lstm
    shared_kalman = KalmanFilter()
    shared_kalman_lstm = KalmanFilterLSTM()
    extents = configs.bird.extents
    rows = configs.bird.rows
    cols = configs.bird.cols
    grid_size_row = (extents[0][1] - extents[0][0]) / rows
    grid_size_col = (extents[1][1] - extents[1][0]) / cols

    def __init__(self, gt_box, gt_id, frame_id, det_box=None, trans_mat=None, det_feat_map=None):
        """
            gt_box: when tracking gt_box is need for every frame
            det_box: some lost detect also need to be track in this frame
            det_feat_map: features map of detection
        """
        self.is_activated = True
        self.frame_ids = []  # frame id of this STrack
        self.det_boxes = []
        self.gt_boxes = []
        self.det_feats = []
        self.trans_matrix = []
        self.gt_id = gt_id  # ground true id offered by nuscenes
        self.num_mature_age = configs.tracker.num_mature_size
        self.age = -1
        self.add_new_frame(gt_box, gt_id, frame_id, det_box, trans_mat, det_feat_map)

    @property
    def is_mature(self):
        """
            @return: when age is greater than threshold of mature age
        """
        return self.age >= self.num_mature_age

    def add_new_frame(self, gt_box, gt_id, frame_id, det_box, trans_mat, det_feat_map):
        """
            Add a new track for this frame
        """
        self.age += 1
        self.frame_ids.append(frame_id)
        self.gt_boxes.append(gt_box)
        self.det_boxes.append(det_box)
        self.trans_matrix.append(trans_mat)
        self.det_feats.append(self.get_det_feat(det_feat_map, det_box))

    def trans_det_center_2_curr_coord(self):
        """
        :brief: trans last det box to curr coord
        @return:
        """
        pass

    def get_det_feat(self, det_feat_map, box):
        """
            Get det box feature, center or ref points
        """
        center_x = (box[0] + self.extents[0][1]) / self.grid_size_row - 0.5
        center_y = (box[1] + self.extents[1][1]) / self.grid_size_col - 0.5
        center_x = center_x.floor().int()
        center_y = center_y.floor().int()
        center_x = torch.clamp(center_x, 0, self.rows - 1)
        center_y = torch.clamp(center_y, 0, self.cols - 1)
        return det_feat_map[:, center_x, center_y].clone()  # need to be check, the position is right ??

    def get_track_feat(self, trans_matrix):
        """
            Computer tracking feature, directly use last detection features
            @todo
                add LSTM to compute the tracking feature
        """
        center = self.det_boxes[-1][:4].detach().cpu().numpy()
        center[3] = 1
        center_trans = trans_matrix.dot(center)
        center_tensor = torch.from_numpy(center_trans).float().to(self.det_boxes[-1].device)
        return torch.cat([self.det_feats[-1], center_tensor[:2], self.det_boxes[-1][[7, 8]].float()])

    def __repr__(self):
        """
            for print
        """
        rep = f"Single Tracker id_{self.gt_id}: tracked for {self.age}" \
              f" frames, last tracked frame_id={self.frame_ids[-1]}"
        return rep


class MTracker(object):
    """
        multi-object tracker
    """
    def __init__(self):
        self.id_to_stracker = {}
        self.trans_matrix = None

    def __len__(self):
        return len(self.id_to_stracker)

    def __getitem__(self, id):
        if id in self.tracking_ids:
            return self.id_to_stracker[id]
        else:
            print(f'No tracker with id = {id}')
            return None

    def update_trans_mat(self, curr_motion):
        """
        :brief: update trans mat
        @return:
        """
        self.trans_matrix = curr_motion

    def set_frame_id(self, frame_id):
        """
            curr frame id
            @brief
            frame_id: curr frame id
        """
        self.frame_id = frame_id

    @property
    def tracking_ids(self):
        tracking_ids_curr = [elem for elem in self.id_to_stracker.keys()
                             if self.id_to_stracker[elem].frame_ids[-1] == (self.frame_id - 1)]
        # return list(self.id_to_stracker.keys())
        return list(tracking_ids_curr)

    @property
    def size(self):
        return len(self.tracking_ids)
        # return len(self.id_to_stracker)

    @property
    def feature_matrix(self):
        """
            get feature matrix for the multi-object tracking
        """
        matrix = torch.stack([self.id_to_stracker[id].get_track_feat(self.trans_matrix) for id in self.tracking_ids], dim=1)
        return matrix

    @property
    def gt_id_matrix(self):
        matrix = torch.from_numpy(np.array(self.tracking_ids))
        return matrix

    def update_tracker(self, id, gt_box, frame_id, det_box, trans_mat, det_feat_map):
        """
            update tracker
        """
        self.id_to_stracker[id].add_new_frame(
            gt_box=gt_box,
            gt_id=id,
            frame_id=frame_id,
            det_box=det_box,
            trans_mat=trans_mat,
            det_feat_map=det_feat_map
        )

    def add_tracker(self, id, stracker):
        self.id_to_stracker[id] = stracker

    def __repr__(self):
        rep = f"Multi-object tracker with [self.size] single trackers (ids: {self.tracking_ids})"
        return rep


class TrackingModel(nn.Module):
    """
        Tracker Model
    """
    def __init__(self):
        super(TrackingModel, self).__init__()
        self.filter_name = configs.tracker.TRACKING_NAMES_INT
        self.batch_trackers = []
        self.batch_size = configs.tracker.train.batch_size
        for _ in range(configs.tracker.train.batch_size):
            self.batch_trackers.append(MTracker())
        self.frame_id = 0
        self.threshold_gt_match_det = configs.tracker.match.threshold_gt_match_det
        self.max_margin_threshold = configs.tracker.Loss.margin_threshold

        self.cost_matrix_pred_list = []
        self.cost_matrix_gt_list = []

        #  MLP(pair) formula(7)
        mlp_pair_list = []
        #  ego trans -> 16 dims
        pre_channel_pair = (32 + 2 + 2 + 32 + 2 + 2)  # 74 ?? feature + egoV + egoYaw + objectV + position
        for k in range(0, configs.tracker.share_rc.__len__()):
            mlp_pair_list.extend([
                nn.Conv2d(pre_channel_pair, configs.tracker.share_rc[k], kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ])
            pre_channel_pair = configs.tracker.share_rc[k]
            if k == configs.tracker.share_rc.__len__() - 1:  # remove last ReLU
                mlp_pair_list.pop()
            if k != configs.tracker.share_rc.__len__() - 1 and configs.tracker.DP_RATIO > 0:
                mlp_pair_list.append(nn.Dropout(configs.tracker.DP_RATIO))
        # mlp_pair_list.append(nn.Sigmoid())  # if softmax not use Sigmoid
        self.mlp_pair = nn.Sequential(*mlp_pair_list).cuda()

        #  MLP(unary) formula(7)
        pre_channel_unary = (32 + 2 + 2)  # 37 ?? feature + egoV + egoYaw + objectV
        mlp_unary_list = []
        for k in range(0, configs.tracker.share_rc.__len__()):
            mlp_unary_list.extend([
                nn.Linear(pre_channel_unary, configs.tracker.share_rc[k], bias=False),
                nn.ReLU(inplace=True)
            ])
            pre_channel_unary = configs.tracker.share_rc[k]  # remove last ReLU
            if k == configs.tracker.share_rc.__len__() - 1:
                mlp_unary_list.pop()
            if k != configs.tracker.share_rc.__len__() - 1 and configs.tracker.DP_RATIO > 0:
                mlp_unary_list.append(nn.Dropout(configs.tracker.DP_RATIO))
        # mlp_unary_list.append(nn.Sigmoid())
        self.mlp_unary = nn.Sequential(*mlp_unary_list).cuda()

    def reset_tracker(self):
        """
            reset_tracker function
        """
        print('reset tracker')
        self.batch_trackers = []
        for _ in range(self.batch_size):
            self.batch_trackers.append(MTracker())
        self.frame_id = 0

    @staticmethod
    def gt_boxes_get_info(gt_boxes_orig):
        """
            some info not save in gt_boxes
            we need extract some info(id, ...) for gt_boxes_orig
            @param gt_boxes_orig: gt boxes origin
            @return: batch_boxes_id
        """
        batch_size = gt_boxes_orig.__len__()
        max_gt = max([x.__len__() for x in gt_boxes_orig])
        batch_boxes_id = np.ones((batch_size, max_gt), dtype=np.int) * -1
        for k in range(batch_size):
            curr_gt_boxes = gt_boxes_orig[k]
            for idx, gt_box in enumerate(curr_gt_boxes):
                batch_boxes_id[k, idx] = gt_box.id
        return batch_boxes_id

    def forward(self, data_dict, det_res_dict, is_training=False):
        """
        @param data_dict: all input data
        @param det_res_dict: all det result data
        @param is_training: training loss
        @return:
        """
        if is_training:
            return self.forward_train(data_dict, det_res_dict)
        else:
            return self.forward_test(data_dict, det_res_dict)

    def build_tracker_feature_matrix(self, curr_gt_match_det_id, batch_idx):
        """
        brify:
            build STrack feature matrix, theoretically we need build velocity diff feature matirx for this cost matrix??
        """
        curr_strack_list = self.tracked_stracks[batch_idx]
        strack_fea_matrix_list = []
        associated_gt_idx = []
        for elem in curr_strack_list:
            if elem.gt_id in curr_gt_match_det_id:
                strack_fea_matrix_list.append(elem.get_recent_fea())
                associated_gt_idx.append(elem.gt_id)
        strack_fea_matrix = torch.stack(strack_fea_matrix_list, 1)
        return strack_fea_matrix, torch.from_numpy(np.array(associated_gt_idx)).cuda()

    def filter_tracking_category(self, cur_gt_box, cur_gt_id, cur_det_box):
        mask = torch.zeros_like(cur_gt_box[:, 9])
        for idx in range(mask.shape[0]):
            if int(cur_gt_box[idx, 9].item()) in self.filter_name:
                mask[idx] = 1
            else:
                mask[idx] = 0
        ok = torch.where(mask > 0.1)
        cur_gt_box = cur_gt_box[ok]
        mask_np = mask.detach().long().cpu().numpy()
        cur_gt_id = cur_gt_id[np.where(mask_np > 0.1)]
        # filter det box
        mask_det = torch.zeros_like(cur_det_box[:, 9])
        for idx in range(mask_det.shape[0]):
            mask_det[idx] = 1 if int(cur_det_box[idx, 9].item()) in self.filter_name else 0
        ok_det = torch.where(mask_det > 0.1)
        cur_det_box = cur_det_box[ok_det]
        return cur_gt_box, cur_gt_id, cur_det_box, ok_det

    def filter_detect_category(self, cur_det_box):
        mask_det = torch.zeros_like(cur_det_box[:, 9])
        for idx in range(mask_det.shape[0]):
            mask_det[idx] = 1 if int(cur_det_box[idx, 9].item()) in self.filter_name else 0
        ok_det = torch.where(mask_det > 0.1)
        cur_det_box = cur_det_box[ok_det]
        return cur_det_box, ok_det

    def forward_train(self, data_dict, det_res_dict):
        """
            brief:
                for train tracker
            Input:
                data_dict: for (gt_boxes, gt_boxes_orig, ego_motion)
                det_res_dict: for (det_boxes, def_fea_maps, box_center_int_pixel)
            info:
                det_boxes: curr detect bboxes
                gt_boxes: curr gt bboxes
                gt_boxes_orig: orig gt_boxes with more info
                batch_det_fea_maps: all det feature map
                box_center_int_pixel: current det box center in pixel coordinate
        """
        self.cost_matrix_pred_list = []
        self.cost_matrix_gt_list = []
        device = det_res_dict['batch_det_feature_map'].device
        batch_size = det_res_dict['batch_det_feature_map'].shape[0]
        gt_boxes = det_res_dict['gt_boxes_torch']
        gt_boxes_orig = data_dict['gt_boxes_orig']  # tensor bs, num_max_batch_box, box_elem
        det_boxes = det_res_dict['bboxes']  # list det box
        # filter some do not need  category
        det_fea_maps = det_res_dict['batch_det_feature_map']
        ego_motions = data_dict['ego_motion']
        det_center_in_pixel = det_res_dict['box_center_in_pixel']
        det_vel_maps = det_res_dict['velocity_pred']

        batch_boxes_id = self.gt_boxes_get_info(gt_boxes_orig)
        batch_size = gt_boxes.shape[0]
        assert batch_size == self.batch_size, "input data batch size not equal to configs batch_size"
        for k in range(batch_size):  # for shi shao shuai method, about iou matrix
            self.batch_trackers[k].set_frame_id(frame_id=self.frame_id)
            curr_ego_motion = ego_motions[k]
            curr_det_fea_map = det_fea_maps[k]
            curr_det_vel_map = det_vel_maps[k]
            cur_gt = gt_boxes[k]
            cur_det = det_boxes[k]
            cur_gt_id = batch_boxes_id[k]
            cur_gt, cur_gt_id, cur_det, filter_det_cat = self.filter_tracking_category(cur_gt, cur_gt_id, cur_det)
            cnt = cur_gt.shape[0] - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_id = cur_gt_id[:cnt + 1]
            if cur_gt.shape[0] * cur_det.shape[0] == 0:
                continue
            ious = iou3d_nms_utils.boxes_iou_bev(cur_gt[:, 0:7].float(), cur_det[:, 0:7].float())

            # compute the matched detection box of each gt box, in order to update the tracker
            max_iou_pre_gt = ious.max(dim=1)
            matched_gt_mask = max_iou_pre_gt.values > self.threshold_gt_match_det
            matched_gt_boxes = cur_gt[matched_gt_mask]
            matched_gt_idx = cur_gt_id[matched_gt_mask.detach().cpu().numpy()]
            matched_det_idx_per_gt = max_iou_pre_gt.indices[matched_gt_mask]
            matched_det_box_per_gt = cur_det[matched_det_idx_per_gt]
            num_matched = matched_gt_mask.sum()  # num of matched track-det pairs

            # compute the matched gt_idx of each detection box, in order to build cost_matrix_gt
            matched_gt_idx_per_det = torch.ones([cur_det.shape[0]]).int() * -1
            for idx in range(num_matched):
                gt_id = matched_gt_idx[idx]
                matched_gt_idx_per_det[matched_det_idx_per_gt[idx]] = int(gt_id)

            # For first frame, only initialize the trackers
            if self.frame_id == 0:
                for idx in range(num_matched):
                    gt_id = matched_gt_idx[idx]
                    self.batch_trackers[k].add_tracker(
                        id=gt_id,
                        stracker=STracker(
                            gt_box=matched_gt_boxes[idx],
                            gt_id=gt_id,
                            frame_id=self.frame_id,
                            det_box=matched_det_box_per_gt[idx],
                            trans_mat=curr_ego_motion,
                            det_feat_map=curr_det_fea_map,
                        )
                    )
                continue
            # if there is no alive tracker for current frame, the skip the loss computation part
            num_track = self.batch_trackers[k].size
            num_det = cur_det.shape[0]
            if num_track * num_det == 0:
                continue
            # build track feature matrix with ego motion
            self.batch_trackers[k].update_trans_mat(curr_ego_motion)
            track_fea_with_velocity = self.batch_trackers[k].feature_matrix

            ego_motion = torch.tensor(curr_ego_motion.reshape(-1)).float().to(device).unsqueeze(dim=1)
            # track_fea_with_velocity = torch.cat([track_feats, ego_motion.repeat(1, num_track)], dim=0)

            # build detect feature matrix with ego motion and object velocity
            center_in_pixel_x = det_center_in_pixel[k][0][filter_det_cat]
            center_in_pixel_y = det_center_in_pixel[k][1][filter_det_cat]
            det_centers = cur_det[:, :2].float().T
            det_feats = curr_det_fea_map[:, center_in_pixel_x, center_in_pixel_y]
            det_velocity = curr_det_vel_map[:, center_in_pixel_x, center_in_pixel_y]
            # det_feats_with_velocity = torch.cat([det_feats, det_centers, det_velocity,
            #                                      ego_motion.repeat(1, num_det)], dim=0)
            det_feats_with_velocity = torch.cat([det_feats, det_centers, det_velocity], dim=0)

            # construct track0det matching matrix
            track_feats_with_velocity_cat_prep = track_fea_with_velocity.unsqueeze(2).repeat(1, 1, num_det)
            det_feats_with_velocity_cat_prep = det_feats_with_velocity.unsqueeze(1).repeat(1, num_track, 1)
            track_dets_feature_concat = torch.cat([det_feats_with_velocity_cat_prep,
                                                   track_feats_with_velocity_cat_prep], dim=0)
            track_dets_feature_concat = track_dets_feature_concat.unsqueeze(0).contiguous()

            # prep target matrix and mlp feature for conv1d
            paired_score = self.mlp_pair(track_dets_feature_concat).squeeze(1)

            # add one more row to handle the false positive detections
            unary_score = self.mlp_unary(det_feats_with_velocity.unsqueeze(dim=0).contiguous().permute(0, 2, 1).float())
            unary_score = unary_score.contiguous().permute(0, 2, 1)
            cost_matrix_pred = torch.cat([paired_score, unary_score], dim=1)

            # prepare gt cost matrix
            tracker_gt_id = self.batch_trackers[k].gt_id_matrix.to(device)
            cost_matrix_gt = tracker_gt_id[:, None] == matched_gt_idx_per_det[None, :].to(device)
            cost_matrix_virtual_label = torch.logical_not(cost_matrix_gt.max(0)[0]).int()
            cost_matrix_gt = torch.cat([cost_matrix_gt.int(), cost_matrix_virtual_label[None]], dim=0)

            # track_dets_fea_with_virtual_result(1, 1, N_Strack + N_virtual, N_Detect) => cost_matrix_gt(N_Strack, N_virtual, N_Detect)
            if not cost_matrix_pred[0].shape == cost_matrix_gt.shape:
                import pdb
                pdb.set_trace()
            self.cost_matrix_pred_list.append(cost_matrix_pred[0])
            self.cost_matrix_gt_list.append(cost_matrix_gt)

            # for frame_id > 1, update the tracker after computing the loss
            for idx in range(num_matched):
                id = matched_gt_idx[idx]
                if id in self.batch_trackers[k].tracking_ids:
                    self.batch_trackers[k].update_tracker(
                        id=matched_gt_idx[idx],
                        gt_box=matched_gt_boxes[idx],
                        frame_id=self.frame_id,
                        det_box=matched_det_box_per_gt[idx],
                        trans_mat=curr_ego_motion,
                        det_feat_map=curr_det_fea_map,
                    )
                else:  # add new single tracker
                    self.batch_trackers[k].add_tracker(
                        id=matched_gt_idx[idx],
                        stracker=STracker(
                            gt_box=matched_gt_boxes[idx],
                            gt_id=matched_gt_idx[idx],
                            frame_id=self.frame_id,
                            det_box=matched_det_box_per_gt[idx],
                            trans_mat=curr_ego_motion,
                            det_feat_map=curr_det_fea_map,
                        )
                    )
        self.frame_id += 1
        return self.get_loss()

    def get_loss(self):
        """
            brief:
                L(score) =  1 / (N(i, j), i belong pos, j belong neg) Sum (max(0, const_margin_threshold - (a(i) - a(j))))
            Using:
                track_det_fea_with_virtual_result_list: all features conv by stract list and detect list
                cost_matrix_gt_list: all features label matrix
            loss function, max margin loss
            Return:
                loss of curr batch struct list and detect list pair and unary
        """
        loss_all_batch = 0
        acc = 0
        for k in range(self.cost_matrix_pred_list.__len__()):
            cost_matrix_pred = self.cost_matrix_pred_list[k]
            cost_matrix_gt = self.cost_matrix_gt_list[k]
            # curr_loss = F.binary_cross_entropy(cost_matrix_pred, cost_matrix_gt.float())
            # positive_x, positive_y = torch.where(cost_matrix_gt)
            # sort_by_det = torch.argsort(positive_y)
            # positive_y = positive_y[sort_by_det]
            # positive_x = positive_x[sort_by_det]
            # formula_aj_diff_ai_add_m = (cost_matrix_pred[:, positive_y] -
            #                              cost_matrix_pred[positive_x, positive_y]) + self.max_margin_threshold
            # # formula_aj_diff_ai_add_m[positive_x, torch.arange(0, formula_aj_diff_ai_add_m.shape[1]).cuda()] = 0.0
            # formula_aj_diff_ai_add_m[positive_x, positive_y] = 0.0
            # curr_loss_total = torch.clamp(formula_aj_diff_ai_add_m, min=0.0).sum()
            # curr_loss = curr_loss_total / (formula_aj_diff_ai_add_m.shape[0] * formula_aj_diff_ai_add_m.shape[1] -
            #                                formula_aj_diff_ai_add_m.shape[0] + 1.0)
            cost_matrix_pred = cost_matrix_pred.softmax(0)
            curr_loss = F.cross_entropy(cost_matrix_pred.T, cost_matrix_gt.argmax(0))

            score_pos_per_det = (cost_matrix_gt * cost_matrix_pred).max(1).values
            score_neg_per_det = ((1 - cost_matrix_gt) * cost_matrix_pred).max(1).values
            curr_acc = (score_pos_per_det > score_neg_per_det).float().mean() * 100
            acc += curr_acc

            loss_all_batch = loss_all_batch + curr_loss
        if self.cost_matrix_pred_list.__len__():
            print("[INFO] curr acc: {}".format(acc / self.cost_matrix_pred_list.__len__()))
        return loss_all_batch

    def get_curr_frame_track(self):
        """
        brief:
            get curr frame track box for vis, batch_size = 1
        @return:
        """
        curr_batch_track = self.batch_trackers[0]
        curr_frame_id = self.frame_id - 1
        res_dict = {}
        track_boxes = []
        track_boxes_age = []
        track_ids = []
        for id in curr_batch_track.tracking_ids:
            curr_track = curr_batch_track.id_to_stracker[id]
            if curr_frame_id not in curr_track.frame_ids:
                continue
            if (self.frame_id >= configs.tracker.num_mature_size) \
                    and (curr_track.age < configs.tracker.num_mature_size):
                continue
            curr_track_box = curr_track.det_boxes[-1]
            track_boxes_age.append(curr_track.age)
            curr_track_id = id
            track_boxes.append(curr_track_box)
            track_ids.append(curr_track_id)
        res_dict.update({'track_box': track_boxes,
                         'track_ids': track_ids,
                         'track_boxes_age': track_boxes_age})
        return res_dict

    def build_target_tracking_matrix_label(self, two_frame_associated_gt_idx,
                                                 curr_det_match_curr_gt_id):
        """
            brify:
                build target tracking matrix label, coor for cost matrix
                    Track \\  Det0,  Det1, Det2, Det3, Det4, Det5, Det6
                    Track A \------------------------------------------
                    Track B \------------------------------------------
                    Track C \------------------------------------------
                    Track D \------------------------------------------
                    Virtual Track \------------------------------------
            Input:
                two_frame_associated_gt_idx : witch gt box id, curr frame and last frame both have
                curr_det_match_curr_gt_id : witch gt box curr det associate with.
            logical:
                curr det -> gt box(curr frame)  <-> gt box(prev frame) -> perv det
            output:
                shape: [num_strack_object, num_detect_object]
                build cost matrix target tracking label matrix
        """
        # num_curr_det_obj = curr_det_match_curr_gt_id.shape[0]
        # num_gt_box_both_curr_prev_have = two_frame_associated_gt_idx.shape[0]
        cost_matrix_gt = (two_frame_associated_gt_idx[:, None] == curr_det_match_curr_gt_id[None]).int()
        return cost_matrix_gt

    def update_forward(self, det_boxes):
        """
            brify:
                for test tracking
            Input:
                det_boxes: curr detect bboxes
        """
        pass

    def forward_test(self, data_dict, det_res_dict):
        """
        @param data_dict:
        @param det_res_dict:
        @return:
        """
        device = det_res_dict['batch_det_feature_map'].device
        batch_size = det_res_dict['batch_det_feature_map'].shape[0]

        det_boxes = det_res_dict['bboxes']  # list det box
        det_feat_maps = det_res_dict['batch_det_feature_map']
        det_center_in_pixel = det_res_dict['box_center_in_pixel']
        det_vel_maps = det_res_dict['velocity_pred']
        ego_motions = data_dict['ego_motion']

        curr_ego_motion = ego_motions
        curr_det_feat_map = det_feat_maps[0]
        curr_det_vel_map = det_vel_maps[0]
        curr_det_box = det_boxes[0]
        curr_det_box, det_filter_idx = self.filter_detect_category(curr_det_box)
        det_center_in_pixel = det_center_in_pixel

        num_det = curr_det_box.shape[0]
        if self.frame_id == 0:
            # for first frame, initialize the tracker
            for idx in range(num_det):
                box_id = len(self.batch_trackers[0])
                self.batch_trackers[0].add_tracker(
                    id=box_id,
                    stracker=STracker(
                        gt_box=None,
                        gt_id=box_id,
                        frame_id=self.frame_id,
                        det_box=curr_det_box[idx],
                        trans_mat=curr_ego_motion,
                        det_feat_map=curr_det_feat_map
                    ))
            self.frame_id += 1
            self.batch_trackers[0].set_frame_id(self.frame_id)
            return self.get_curr_frame_track()

        num_track = self.batch_trackers[0].size
        num_det = curr_det_box.shape[0]

        self.batch_trackers[0].update_trans_mat(curr_ego_motion)
        track_feats_with_velocity = self.batch_trackers[0].feature_matrix
        # ego_motion = torch.tensor(curr_ego_motion.reshape(-1)).float().to(device).unsqueeze(dim=1)
        # track_feats_with_velocity = torch.cat([track_feats, ego_motion.repeat(1, num_track)], dim=0)

        #  build detect feature matrix with ego motion and object velocity
        center_in_pixel_x = det_center_in_pixel[0][0][det_filter_idx]
        center_in_pixel_y = det_center_in_pixel[0][1][det_filter_idx]
        det_centers = curr_det_box[:, :2].float().T
        det_feats = curr_det_feat_map[:, center_in_pixel_x, center_in_pixel_y]
        det_velocity = curr_det_vel_map[:, center_in_pixel_x, center_in_pixel_y]
        det_feats_with_velocity = torch.cat([det_feats, det_centers, det_velocity], dim=0)
        # construct track-det matching matrix
        track_feats_with_velocity_cat_prep = track_feats_with_velocity.unsqueeze(2).repeat(1, 1, num_det)
        det_feats_with_velocity_cat_prep = det_feats_with_velocity.unsqueeze(1).repeat(1, num_track, 1)
        track_dets_feature_concat = torch.cat([det_feats_with_velocity_cat_prep,
                                               track_feats_with_velocity_cat_prep], dim=0)
        track_dets_feature_concat = track_dets_feature_concat.unsqueeze(0).contiguous()

        #  prep target matrix and mlp feature for conv1d
        paired_score = self.mlp_pair(track_dets_feature_concat).squeeze(1)
        if num_det * num_track == 0:
            self.frame_id += 1
            self.batch_trackers[0].set_frame_id(self.frame_id)
            return self.get_curr_frame_track()

        #  add one more row to handle the false positive detections.
        unary_score = self.mlp_unary(det_feats_with_velocity.unsqueeze(dim=0).contiguous().permute(0, 2, 1).float())
        unary_score = unary_score.contiguous().permute(0, 2, 1)
        cost_matrix_pred = torch.cat([paired_score, unary_score], dim=1)  # [n_track + 1, n_det]

        # print("-------------------------------------------------------------")
        # print(cost_matrix_pred[0].detach().cpu().numpy())
        # print("*************************************************************")
        # hungarian algorithm
        cost_matrix_pred_softmax = cost_matrix_pred[0].softmax(0)
        matched_indices, unmatched_tracker, newborn_det_indices = linear_assignment(  # unmatched_tracker not use
            cost_matrix=(1.0 - cost_matrix_pred_softmax.detach().cpu().numpy()), thresh=0.6)

        #  for matched track-det, update the tracker
        tracking_ids_tmp = self.batch_trackers[0].tracking_ids.copy()
        for tracker_idx, det_idx in matched_indices:
            box_id = tracking_ids_tmp[tracker_idx]
            self.batch_trackers[0].update_tracker(
                id=box_id,
                gt_box=None,
                frame_id=self.frame_id,
                det_box=curr_det_box[det_idx],
                trans_mat=curr_ego_motion,
                det_feat_map=curr_det_feat_map
            )

        #  for newborn det, initialize the tracker
        for det_idx in newborn_det_indices:
            box_id = len(self.batch_trackers[0])
            self.batch_trackers[0].add_tracker(
                id=box_id,
                stracker=STracker(
                    gt_box=None,
                    gt_id=box_id,
                    frame_id=self.frame_id,
                    det_box=curr_det_box[det_idx],
                    trans_mat=curr_ego_motion,
                    det_feat_map=curr_det_feat_map
                )
            )
        print("Frame {}: current tracker={} newborn tracker={}"
              .format(self.frame_id, len(self.batch_trackers[0]), len(newborn_det_indices)))
        self.frame_id += 1
        self.batch_trackers[0].set_frame_id(self.frame_id)
        return self.get_curr_frame_track()










