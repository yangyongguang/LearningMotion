import torchvision
import torch
import torch.nn as nn
import configs
from model import MotionNet
from model import UNet

from data.nuscenes_dataloader import TrainDatasetMultiSeq

model_path = "/media/yyg/C14D581BDA18EBFA/code/MotionNet/logs/train_multi_seq/det_corret_voxel_count_gt_re_train/epoch_5.pth"

data = torch.rand(1, 3, 512, 512)

resume_det = "/media/yyg/C14D581BDA18EBFA/code/MotionNet/logs/train_multi_seq/box_512_baseline/epoch_12.pth"
# resume_det = "/media/yyg/C14D581BDA18EBFA/code/MotionNet/logs/train_multi_seq/det_da_6epoch_0724/epoch_6.pth"
resume_tracking = "/media/yyg/C14D581BDA18EBFA/code/MotionNet/checkpoint/tracking_ass_new_motion/epoch_7.pth"

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_nuscenes_base = TrainDatasetMultiSeq(batch_size=1, devices=devices, tracking=True, split='val', vispy_seq=True)
data_nuscenes = TrainDatasetMultiSeq(batch_size=1, devices=devices, tracking=True, split='val', vispy_seq=True)
# trainloader = torch.utils.data.DataLoader(data_nuscenes,
#                                           batch_size=configs.data.batch_size,
#                                           shuffle=False,
#                                           num_workers=configs.data.num_worker,
#                                           collate_fn=data_nuscenes.collate_batch)

model_detect = MotionNet(num_feature_channel=configs.bird.num_feature_channel, batch_size=2, device=devices, is_training=False)
model_detect = nn.DataParallel(model_detect)
model_detect = model_detect.to(devices)
# checkpoint = torch.load(resume_det)
#
# data_dict = data_nuscenes[0]
# data_dict_add_base = data_nuscenes_base[0]
# batch_list_data = [data_dict, data_dict_add_base]
# data_dict_input = TrainDatasetMultiSeq.collate_batch(batch_list_data)
#
# # model_detect.load_state_dict(checkpoint['model_state_dict'])
#
# det_res_dict = model_detect(data_dict_input)

onnx_path = "/media/yyg/C14D581BDA18EBFA/code/MotionNet/logs/" \
            "train_multi_seq/det_corret_voxel_count_gt_re_train/model.onnx"

# torch.onnx.export(model_detect, data_dict_input, onnx_path)

data = torch.rand(1, 80, 512, 512)
unet = UNet(n_channels=configs.bird.num_feature_channel)
output = unet(data)
torch.onnx.export(unet, data, onnx_path)






