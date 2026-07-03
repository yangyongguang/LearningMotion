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


def main():
    start_epoch = 1
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)
    torch.multiprocessing.set_start_method("spawn")
    data_nuscenes = TrainDatasetMultiSeq(devices=devices, tracking=True)
    need_shuffle = not configs.tracker
    trainloader = torch.utils.data.DataLoader(data_nuscenes, batch_size=configs.data.batch_size, shuffle=need_shuffle,
                                              num_workers=configs.data.num_worker, collate_fn=data_nuscenes.collate_batch)
    # for tracking set is trainging for false, when not traing with end to end
    model = MotionNet(num_feature_channel=configs.bird.num_feature_channel, device=devices, is_training=False)
    model = nn.DataParallel(model)
    model = model.to(devices)

    checkpoint = torch.load(configs.train.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # freeze detect model param
    for param in model.parameters():
        param.requires_grad = False

    eval(model, trainloader)
    debug = 1


def eval(model, trainloader):
    for i, data_dict in enumerate(trainloader, 0):
        scene_idxes = data_dict['scene_idx']
        sweep_idxes = data_dict['sweep_idx']
        # gen detect bbox for training
        det_res_dict = model(data_dict)

        confidence_pred = det_res_dict['confidence_pred']
        voxel_count_gt = data_dict['voxel_count_map']
        blocking_pred = det_res_dict['blocking_pred']
        offset_pred = det_res_dict['offset_pred']
        blocking_gt = data_dict['blocking_target_map']
        data_idx = i
        # voxel_count_gt, blocking_pred, offset_pred, blocking_gt
        torch.sigmoid(confidence_pred).detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/"
            "simpleViewer/binFile/confidence_pred_{}.bin".format(
            data_idx))
        voxel_count_gt.detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "voxel_count_gt_{}.bin".format(data_idx))
        torch.sigmoid(blocking_pred).detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/"
            "binFile/blocking_pred_{}.bin".format(data_idx))
        blocking_gt.detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "blocking_gt_{}.bin".format(data_idx))
        offset_pred.detach().cpu().contiguous().numpy().tofile(
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "offset_pred_{}.bin".format(data_idx))
        offset_pred.detach().cpu().contiguous().numpy().tofile(   #  save twice
            "/media/yyg/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "offset_pred_{}.bin".format(data_idx))
        debug = 1

    loss = 0
    return loss


if __name__ == "__main__":
    main()