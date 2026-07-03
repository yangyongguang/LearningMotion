import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import configs
import time
import sys
import os
from shutil import copytree, copy
from model import MotionNet
from data.nuscenes_dataloader import TrainDatasetMultiSeq

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


need_log = configs.train.log
num_epochs = configs.train.num_epochs


# def main():
#     start_epoch = 1
#     devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.multiprocessing.set_start_method("spawn")
#     device_num = torch.cuda.device_count()
#     print("device number", device_num)
#     data_nuscenes = TrainDatasetMultiSeq(devices=devices, tracking=True)
#     need_shuffle = not configs.tracker
#     trainloader = torch.utils.data.DataLoader(data_nuscenes, batch_size=configs.data.batch_size, shuffle=need_shuffle,
#                                               num_workers=configs.data.num_worker, collate_fn=data_nuscenes.collate_batch)
#     # for tracking set is trainging for false, when not traing with end to end
#     #  for model load checkpoint
#     model = MotionNet(num_feature_channel=configs.bird.num_feature_channel, device=devices, is_training=False)
#     model = nn.DataParallel(model)
#     model = model.to(devices)
#     checkpoint = torch.load(configs.val.model_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     for i, data_dict in enumerate(trainloader, 0):
#         det_res_dict = model(data_dict)
#         debug = 1
#
# if __name__ == "__main__":
#     main()
#
# sys.exit(0)


import numpy as np
from ops.pixel_cluster import pixel_cluster_utils
from ops.rtree import rtree_utils
from ops.roiaware_pool3d import roiaware_pool3d_utils
from nuscenes.utils.data_classes import Box


import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import configs
import time
import sys
import os
from shutil import copytree, copy
from model import MotionNet
from data.nuscenes_dataloader import TrainDatasetMultiSeq
from pyquaternion import Quaternion
from utils.common_utils import RANDOR_COLORS
import matplotlib.pyplot as plt

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#  for vis import
# import vispy
# from vispy import scene
# from vispy import app
# from vispy.io import load_data_file, read_png
# import matplotlib.pyplot as plt
# from vispy import app, visuals
# from vispy.scene import visuals, SceneCanvas
# from vispy.scene.visuals import Text
# from vispy.visuals import TextVisual


# for data prep
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_num = torch.cuda.device_count()
print("device number", device_num)
torch.multiprocessing.set_start_method("spawn")
data_nuscenes = TrainDatasetMultiSeq(devices=devices, tracking=True)
print("Training dataset size:", len(data_nuscenes))

#  for model load checkpoint
model = MotionNet(num_feature_channel=configs.bird.num_feature_channel, device=devices, is_training=False)
model = nn.DataParallel(model)
model = model.to(devices)
checkpoint = torch.load(configs.val.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# if __name__ == "__main__":
#     vistool = vis(offset=6466)
#     # vistool = vis(offset=79882)
#     # vistool = vis(offset=2402)
#     vispy.app.run()
#
# sys.exit(0)



# #  get one prediction
# data_dict = data_nuscenes[79957]
# data_dict = data_nuscenes[81150]
# data_idx = 27666
data_idx_0 = 79982
data_idx2 = 11

need_shuffle = False
# trainloader = torch.utils.data.DataLoader(data_nuscenes, batch_size=configs.data.batch_size, shuffle=need_shuffle,
#                                           num_workers=configs.data.num_worker, collate_fn=data_nuscenes.collate_batch)
#
# for idx, data_dict in enumerate(trainloader):
#     det_res_dict = model(data_dict)
#     debug = 1

# sys.exit(0)
fig, ax = plt.subplots(1, 2, figsize=(50, 20))

i = 0
for data_idx in range(data_idx_0, data_idx_0 + 3000, 1):
    data_dict = data_nuscenes[data_idx]
    # data_idx2 = 75516
    data_dict2 = data_nuscenes[data_idx2]

    all_pc = data_dict['points'].detach().cpu().numpy()
    blocking_gt = data_dict['blocking_target_map']
    offset_gt = data_dict['offset_target_map']
    velocity_gt = data_dict['velocity_target_map']
    voxel_count_gt = data_dict['voxel_count_map']

    all_pc2 = data_dict2['points'].detach().cpu().numpy()
    blocking_gt2 = data_dict2['blocking_target_map']
    offset_gt2 = data_dict2['offset_target_map']
    velocity_gt2 = data_dict2['velocity_target_map']
    voxel_count_gt2 = data_dict2['voxel_count_map']

    voxel_feature = torch.cat([data_dict['voxel_feature'], data_dict2['voxel_feature']], dim=0)
    # voxel_feature = data_dict['voxel_feature']
    #  for blocking loss
    model_res_dict = model(data_dict)
    blocking_pred = model_res_dict['blocking_pred']
    confidence_pred = model_res_dict['confidence_pred']
    velocity_pred = model_res_dict['velocity_pred']
    offset_pred = model_res_dict['offset_pred']

    blocking_pred = blocking_pred.view(-1, configs.bird.rows, configs.bird.cols).contiguous()

    voxel_count_gt = torch.cat([voxel_count_gt, voxel_count_gt2], dim=0)
    blocking_gt = torch.cat([blocking_gt, blocking_gt2], dim=0)

    confidence_pred = confidence_pred.view(-1, configs.bird.rows, configs.bird.cols).contiguous()
    pixel_cluster = pixel_cluster_utils.PixelCluster(isTrain=True)
    blocking_weight, _ = pixel_cluster(voxel_count_gt.cpu(), torch.sigmoid(blocking_pred).cpu(),
                                       offset_pred.cpu(), blocking_gt.cpu())

    # rtree blocking loss
    batch_size = voxel_count_gt.shape[0]
    rows = configs.bird.rows
    cols = configs.bird.cols
    fixedChannel = 7
    rtree = rtree_utils.RTree(rows=configs.bird.rows, cols=configs.bird.cols,
                              devices=devices, isTrain=True, batch_size=batch_size, numChannel=7)

    blocking_weight_rtree, _, _ = rtree(voxel_count_gt, torch.sigmoid(blocking_pred),
                                        torch.sigmoid(confidence_pred), offset_pred, blocking_gt)

    # for val in rtree
    rtree_val = rtree_utils.RTree(rows=configs.bird.rows, cols=configs.bird.cols,
                                  devices=devices, isTrain=False, batch_size=batch_size, numChannel=7)
    boxes_result_map = rtree_val(voxel_count_gt, torch.sigmoid(blocking_pred),
                                 torch.sigmoid(confidence_pred), offset_pred, blocking_gt=None)

    # torch.cuda.synchronize()
    # plt.figure(figsize=(20, 20))
    # blocking_weight = blocking_weight.detach().numpy()
    # plt.imshow(blocking_weight[0, :, :])
    # #
    # plt.figure(figsize=(20, 20))
    # plt.imshow(blocking_weight_rtree.detach().cpu().numpy()[0, :, :])
    #
    # plt.figure(figsize=(20, 20))
    # plt.imshow((torch.sigmoid(blocking_pred) > 0.5).detach().cpu().numpy()[1, :, :])
    # plt.show()

    fig.tight_layout()
    # fig = plt.figure(figsize=(50, 50))
    rows = configs.bird.rows
    cols = configs.bird.cols
    ax[0].scatter(all_pc[:,  0], all_pc[:, 1], c='grey')
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    ax[0].title.set_text('Lidar data')
    # #  render bbox
    # bboxes = model_res_dict['bboxes'][0]
    # for idx in range(bboxes.shape[0]):
    #     inst = bboxes[idx, :].detach().cpu().numpy()
    #     if np.isnan(inst).any():
    #         continue
    #     size_val = [inst[4], inst[3], inst[5]]
    #     box = Box(center=inst[0:3], size=size_val, orientation=Quaternion(axis=[0, 0, 1], angle=inst[6]))
    #     box.render(ax[0], colors=('r', 'r', 'r'))
    #
    # boxes_gt = data_dict['gt_boxes_orig']
    # for idx in range(boxes_gt.__len__()):
    #     # inst = boxes_gt[0]
    #     # if np.isnan(inst).any():
    #     #     continue
    #     # size_val = [inst[4], inst[3], inst[5]]
    #     # box = Box(center=inst[0:3], size=size_val, orientation=Quaternion(axis=[0, 0, 1], angle=inst[6]))
    #     box = boxes_gt[idx]
    #     box.render(ax[0], colors=('k', 'k', 'k'))

    ax[0].set_xlim([configs.bird.extents[0][0], configs.bird.extents[0][1]])
    ax[0].set_ylim([configs.bird.extents[1][0], configs.bird.extents[1][1]])
    #  draw blocking pred
    confidence_pred_numpy = torch.sigmoid(confidence_pred.view(-1, rows, cols).contiguous()).detach().cpu().numpy()[0, :, :]
    blocking_mask = (confidence_pred_numpy > 0.7)
    idx_x = np.arange(rows)
    idx_y = np.arange(cols)
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    X = idx_x[blocking_mask]
    Y = idx_y[blocking_mask]
    # ax[0].plot(X, Y, '.')
    # ax[0].set_aspect('equal')
    # ax[0].axis('off')
    # ax[0].title.set_text('confidence_pred')
    ax[1].plot(X, Y, '.')
    ax[1].axis('off')
    ax[1].title.set_text('confidence_pred')
    # plt.figure(figsize=(20, 20))
    # pos_x, pos_y = (blocking_gt.detach().cpu().numpy()[0, :, :] > 0.1)
    # plt.draw()
    #
    # plt.figure(figsize=(20, 20))
    #

    boxes_result_map_numpy = boxes_result_map.detach().cpu().numpy()[0, :, :]
    boxes_result_map_numpy_mask = (boxes_result_map_numpy > 0)
    X = idx_x[boxes_result_map_numpy_mask]
    Y = idx_y[boxes_result_map_numpy_mask]
    # ax[0].plot(X, Y, '.', c='r')
    # ax[0].set_aspect('equal')
    # ax[0].axis('off')
    # ax[0].title.set_text('boxes_result_map_result')
    ax[1].plot(X, Y, '.', c='r')
    ax[1].axis('off')

    # velocity_pred
    grid_size = (configs.bird.extents[0][1] - configs.bird.extents[0][0]) / configs.bird.rows
    velocity_pred_numpy = velocity_pred.detach().cpu().numpy()[0]
    X = idx_x[boxes_result_map_numpy_mask]
    Y = idx_y[boxes_result_map_numpy_mask]
    U = velocity_pred_numpy[0, :, :][boxes_result_map_numpy_mask] / grid_size
    V = velocity_pred_numpy[1, :, :][boxes_result_map_numpy_mask] / grid_size
    # ax[0].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
    # ax[0].set_aspect('equal')
    # ax[0].title.set_text('velocity Pred')
    # ax[0].axis('off')
    ax[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
    ax[1].axis('off')
    ax[1].set_xlim([0, configs.bird.rows])
    ax[1].set_ylim([0, configs.bird.cols])

    file_name = os.path.join("img_box/", str(i) + '_box.png')
    i += 1
    plt.savefig(file_name)
    print("[INFO] save {} finished".format(file_name))
    # plt.draw()
    # plt.pause(1)
    ax[0].clear()
    ax[1].clear()
# plt.show()
#
#
# plt.show()

# plt.figure(figsize=(20, 20))
# plt.imshow(np.logical_not(fixedMem[1, 2, :, :].detach().cpu().numpy()))
# plt.show()

num_test = 100
# # torch.cuda.synchronize()
# start = time.time()
# for idx in range(num_test):
#     print("[rtree] has pasted {} times".format(idx))
#     blocking_weight_rtree, _ = rtree(voxel_count_gt, blocking_pred, offset_pred, blocking_gt)
# # torch.cuda.synchronize()
# print("[INFO] rtree has cost about: {} ms".format((time.time() - start) * 1000 / num_test))

'''
if __name__ == "__main__":
    # plt.figure(figsize=(20, 20))

    # for data_idx in range(178858):
    for data_idx in range(6057, 6059):
        data_dict = data_nuscenes[data_idx]
        all_pc = data_dict['points'].detach().cpu().numpy()
        blocking_gt = data_dict['blocking_target_map']
        offset_gt = data_dict['offset_target_map']
        velocity_gt = data_dict['velocity_target_map']
        voxel_count_gt = data_dict['voxel_count_map']
        voxel_feature = data_dict['voxel_feature']

        blocking_pred, offset_pred, confidence_pred, velocity_pred = model(voxel_feature)
        # plt.imshow(blocking_weight_rtree.detach().cpu().numpy()[0, :, :])
        # plt.savefig(os.path.join("img/", str(data_idx) + '.png'))

        torch.sigmoid(confidence_pred[0]).detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/"
                                                                                     "simpleViewer/binFile/confidence_pred_{}.bin".format(
            data_idx))
        voxel_count_gt[0].detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "voxel_count_gt_{}.bin".format(data_idx))
        torch.sigmoid(blocking_pred[0]).detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/"
            "binFile/blocking_pred_{}.bin".format(data_idx))
        offset_pred[0].detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/simpleViewer/"
                                                                  "binFile/offset_pred_{}.bin".format(data_idx))
        blocking_gt[0].detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "blocking_gt_{}.bin".format(data_idx))
        print("[rtree] has pasted {} times".format(data_idx))
        start = time.time()
        # rtree = rtree_utils.RTree(rows=configs.bird.rows, cols=configs.bird.cols,
        #                           devices=devices, isTrain=True, batch_size=1, numChannel=7)
        # blocking_weight_rtree, _ = rtree(voxel_count_gt, blocking_pred, offset_pred, blocking_gt)

        # for blocking loss
        pixel_cluster = pixel_cluster_utils.PixelCluster(isTrain=True)
        blocking_weight, _ = pixel_cluster(voxel_count_gt.cpu(), blocking_pred.cpu(),
                                           offset_pred.cpu(), blocking_gt.cpu())

        torch.cuda.synchronize()
        print("[INFO] rtree has cost about: {} ms".format((time.time() - start) * 1000))
        torch.cuda.synchronize()
        # if data_idx > num_test:
        #     break

'''
# plt.figure(figsize=(20, 20))
# blocking_weight = blocking_weight.detach().numpy()
# plt.imshow(blocking_weight[0, :, :])
# # #
# plt.figure(figsize=(20, 20))
# plt.imshow(blocking_weight_rtree.detach().cpu().numpy()[0, :, :])
# plt.show()
# #
# # plt.figure(figsize=(20, 20))
# # plt.imshow((torch.sigmoid(blocking_pred) > 0.5).detach().cpu().numpy()[1, :, :])
# # plt.show()
#
# plt.figure(figsize=(20, 20))
# plt.imshow(blocking_gt.detach().cpu().numpy()[0, :, :])
# plt.show()

# plt.figure(figsize=(20, 20))
# plt.imshow(np.logical_not(fixedMem[1, 2, :, :].detach().cpu().numpy()))
# plt.show()

sys.exit(0)
class vis():
    def __init__(self, offset=0):
        self.offset = offset
        self.play = False
        self.reset()
        self.to_next_sample()

    def reset(self):
        self.action = "no" #, no , next, back, quit
        self.canvas = SceneCanvas(keys='interactive', size=(1600, 1200), show=True, bgcolor='k')
        self.canvas.events.key_press.connect(self.key_press_event)

        # interface(n next, b back, q quit, very simple)
        self.lidar_view = self.canvas.central_widget.add_view()
        self.lidar_view.camera = 'turntable'
        visuals.XYZAxis()
        self.lidar_vis = visuals.Markers()
        self.lidar_view.add(self.lidar_vis)
        self.extents = np.array(configs.bird.extents)
        self.rows = configs.bird.rows
        self.cols = configs.bird.cols
        self.coor = np.array([self.extents[0, 0], self.extents[1, 0]])
        self.resolution = configs.bird.resolution
        # draw lidar boxes
        # self.line_vis = visuals.Line(color='r', method='gl', connect="segments", name="boxes line")
        self.velocity_vis = visuals.Line(color='r', method='gl', connect="segments", name="velocity line")
        # self.lidar_view.add(self.line_vis)
        self.lidar_view.add(self.velocity_vis)

    def to_next_sample(self):
        print("[to_netxt_sample] [" + str(self.offset) + "] to_next_sample")
        data_dict = data_nuscenes[self.offset]
        pts = data_dict['points'].detach().cpu().numpy()
        extents = self.extents
        # prep confidence pred
        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]

        pts = pts[filter_idx]
        num_pts = pts.shape[0]
        blocking_pred, offset_pred, confidence_pred, velocity_pred = model(data_dict['voxel_feature'])
        confidence_pred_np = torch.sigmoid(confidence_pred).view(-1, self.rows,
                                                                 self.cols).contiguous().detach().cpu().numpy()[0, :, :]

        blocking_pred_np = torch.sigmoid(blocking_pred).view(-1, self.rows,
                                                                 self.cols).contiguous().detach().cpu().numpy()[0, :, :]
        pos_2d = np.round((pts[:, :2] - self.coor - self.resolution * 0.5) / self.resolution).astype(np.int)
        pos_2d = np.clip(pos_2d, a_min=0, a_max=255)
        colors = np.ones((num_pts, 4), dtype=np.float)
        ## show lidar
        is_fg = np.where(np.logical_and(confidence_pred_np[pos_2d[:, 0], pos_2d[:, 1]] > 0.5,
                                        blocking_pred_np[pos_2d[:, 0], pos_2d[:, 1]] > 0.5))
        colors[is_fg] = [0.0, 1.0, 1.0, 1.0]
        self.lidar_vis.set_gl_state('translucent', depth_test=False)
        self.lidar_vis.set_data(pts[:, :3],
                                edge_color=None, face_color=colors, size=2)
        self.lidar_view.add(self.lidar_vis)
        self.lidar_view.camera = 'turntable'

        # self.draw_boxes(boxes)
        ## next to sample data

    def update_scenes_choose(self):
        self.to_next_sample()


    def draw_boxes(self, boxes):
        num_pts_pre_boxes=24
        all_bboxes_pts = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        box_idx = 0
        color_lines = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        velocity_pts = []
        for box in boxes:
            box_pt = box.corners().transpose()
            all_bboxes_pts[box_idx * num_pts_pre_boxes : (box_idx + 1) * num_pts_pre_boxes, :] =\
                box_pt[[0, 1, 4, 5, 7, 6, 3, 2, 0, 3, 3, 7, 7, 4, 4, 0, 2, 6, 6, 5, 5, 1, 1, 2], :]
            if not np.isnan(box.velocity).any():
                # show velocity
                center = box.center
                target = box.center + box.velocity
                velocity_pts.append(center)
                velocity_pts.append(target)
                # all_bboxes_pts = np.vstack([all_bboxes_pts, center, target])
            color_lines[box_idx * num_pts_pre_boxes : (box_idx + 1) * num_pts_pre_boxes, :] = \
                RANDOR_COLORS[box.id % RANDOR_COLORS.shape[0]] / 255.0
            box_idx += 1
        self.line_vis.set_data(all_bboxes_pts, color=color_lines)
        if velocity_pts.__len__() > 0:
            self.velocity_vis.set_data(np.array(velocity_pts))
        else:
            self.velocity_vis.set_data(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def key_press_event(self, event):
        if event.key == 'N':
            self.offset += 1
            # self.update_canvas()
            self.to_next_sample()
        elif event.key == "B":
            self.offset -= 1
            print("[EVENT] B")
            # self.update_canvas()
            self.to_next_sample()
        elif event.key == "H":
            self.scenes_idx += 1
            if self.scenes_idx >= 849:
                self.scenes_idx = 849
            self.offset = 0
            print("Processing scene {} ...".format(self.scenes_idx))
            self.lastSweepBoxes.clear()
            self.update_scenes_choose()
        elif event.key == "G":
            self.scenes_idx -= 1
            if self.scenes_idx <= 0:
                self.scenes_idx = 0
            self.offset = 0
            print("Processing scene {} ...".format(self.scenes_idx))
            self.lastSweepBoxes.clear()
            self.update_scenes_choose()

    def on_draw(selfself, event):
        print("[KEY INFO] draw")


# ---------------------------------------------------------------------------------
##
# #  draw img
fig, ax = plt.subplots(3, 4, figsize=(50, 50))
data_idx = 11
# voxel_count_gt, blocking_pred, offset_pred, blocking_gt
torch.sigmoid(confidence_pred[1]).detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/"
                                                                "simpleViewer/binFile/confidence_pred_{}.bin".format(data_idx))
voxel_count_gt[1].detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
                                                "voxel_count_gt_{}.bin".format(data_idx))
torch.sigmoid(blocking_pred[1]).detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/simpleViewer/"
                                                              "binFile/blocking_pred_{}.bin".format(data_idx))
offset_pred[1].detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/simpleViewer/"
                                             "binFile/offset_pred_{}.bin".format(data_idx))
blocking_gt[1].detach().cpu().contiguous().numpy().tofile("/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
                                             "blocking_gt_{}.bin".format(data_idx))

# sys.exit(0)

#
#  draw points
ax[0, 0].scatter(all_pc[:, 0], all_pc[:, 1])
ax[0, 0].axis('off')
ax[0, 0].set_aspect('equal')
ax[0, 0].title.set_text('Lidar data')
rows = configs.bird.rows
cols = configs.bird.cols
#  draw blocking pred
blocking_pred_numpy = torch.sigmoid(blocking_pred.view(-1, rows, cols).contiguous()).detach().cpu().numpy()[0, :, :]
blocking_mask = (blocking_pred_numpy > 0.5)
idx_x = np.arange(rows)
idx_y = np.arange(cols)
idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
# X = idx_x[blocking_mask]
# Y = idx_y[blocking_mask]
ax[0, 1].imshow(blocking_mask.T)
# ax[1, 0].plot(X, Y, '.')
ax[0, 1].set_aspect('equal')
ax[0, 1].axis('off')
ax[0, 1].title.set_text('blocking pred')

blocking_gt_numpy = data_dict['blocking_target_map'].detach().cpu().numpy().squeeze()
blocking_gt_mask = (blocking_gt_numpy > 0.5)
# X = idx_x[blocking_mask]
# Y = idx_y[blocking_mask]
ax[0, 2].imshow(blocking_gt_mask.T)
# ax[1, 0].plot(X, Y, '.')
ax[0, 2].set_aspect('equal')
ax[0, 2].axis('off')
ax[0, 2].title.set_text('blocking gt')

#  draw confidence pred
confidence_pred_numpy = torch.sigmoid(confidence_pred.view(-1, rows, cols).contiguous()).detach().cpu().numpy()[0, :, :]
confidence_mask = (confidence_pred_numpy > 0.6)
idx_x = np.arange(rows)
idx_y = np.arange(cols)
idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
# X = idx_x[confidence_mask]
# Y = idx_y[confidence_mask]
ax[0, 3].imshow(confidence_mask.T)
# ax[1, 0].plot(X, Y, '.')
ax[0, 3].set_aspect('equal')
ax[0, 3].axis('off')
ax[0, 3].title.set_text('confidence pred')

#  draw offset pred
pos_nonzeros_tuple = np.where(blocking_gt_mask == True)
#  too mush positive pixel need to show, we sample it
pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
pos_selected_mask = np.zeros_like((blocking_gt_numpy > 0.5)).astype(np.bool)
pos_selected_mask[pox_x, pos_y] = True
offset_pred_map_numpy = offset_pred.detach().cpu().numpy().squeeze()
X = idx_x[pos_selected_mask]
Y = idx_y[pos_selected_mask]
U = -offset_pred_map_numpy[0, :, :][pos_selected_mask]
V = -offset_pred_map_numpy[1, :, :][pos_selected_mask]
ax[1, 0].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
ax[1, 0].set_aspect('equal')
ax[1, 0].title.set_text('offset Prediction')
ax[1, 0].axis('off')

#  draw offset gt
pos_nonzeros_tuple = np.where(blocking_gt_mask == True)
#  too mush positive pixel need to show, we sample it
pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
pos_selected_mask = np.zeros_like((blocking_gt_numpy > 0.5)).astype(np.bool)
pos_selected_mask[pox_x, pos_y] = True
offset_target_map_numpy = data_dict['offset_target_map'].detach().cpu().numpy().squeeze()
X = idx_x[pos_selected_mask]
Y = idx_y[pos_selected_mask]
U = -offset_target_map_numpy[0, :, :][pos_selected_mask]
V = -offset_target_map_numpy[1, :, :][pos_selected_mask]
ax[1, 1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
ax[1, 1].set_aspect('equal')
ax[1, 1].title.set_text('offset ground truth')
ax[1, 1].axis('off')

ax[1, 2].imshow(blocking_weight.detach().cpu().numpy()[0].T)
ax[1, 2].set_aspect('equal')
ax[1, 2].axis('off')
ax[1, 2].title.set_text('blocking weight')

#  draw velocity pred
grid_size = (configs.bird.extents[0][1] - configs.bird.extents[0][0]) / configs.bird.rows
pos_selected_mask = np.zeros_like((blocking_gt_numpy > 0.5)).astype(np.bool)
pos_selected_mask[pox_x[::5], pos_y[::5]] = True
velocity_pred_map_numpy = velocity_pred.detach().cpu().numpy().squeeze()
X = idx_x[pos_selected_mask]
Y = idx_y[pos_selected_mask]
U = velocity_pred_map_numpy[0, :, :][pos_selected_mask] / grid_size
V = velocity_pred_map_numpy[1, :, :][pos_selected_mask] / grid_size
ax[2, 0].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
ax[2, 0].set_aspect('equal')
ax[2, 0].title.set_text('velocity Prediction')
ax[2, 0].axis('off')

#  draw velocity pred
velocity_target_map_numpy = data_dict['velocity_target_map'].detach().cpu().numpy().squeeze()
U = velocity_target_map_numpy[0, :, :][pos_selected_mask] / grid_size
V = velocity_target_map_numpy[1, :, :][pos_selected_mask] / grid_size
ax[2, 1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
ax[2, 1].set_aspect('equal')
ax[2, 1].title.set_text('velocity gt')
ax[2, 1].axis('off')

plt.show()
# plt.savefig(os.path.join("img/", str(idx) + '.png'))
ax[0, 0].clear()
ax[0, 1].clear()
ax[0, 2].clear()
ax[1, 0].clear()
ax[1, 1].clear()
ax[1, 2].clear()
ax[2, 0].clear()
ax[2, 1].clear()

plt.close()






