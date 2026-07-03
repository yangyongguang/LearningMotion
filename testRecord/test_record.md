## test record 

### test 1, confidence 0.06
```
mAP => 0.1913
```

### test 2, fix confidence threshold
```
mAP: 0.2702 mATE: 0.4585 mASE: 0.3528 mAOE: 1.6209 mAVE: 0.4922 mAAE: 0.2286 NDS: 0.3819
```

### test 3, fix voxel count gt for fine tune 5-6 epoch
```

```

### test 4, add weight attribute for center point
```

```

```
pos_x, pos_y = torch.where(boxes_result_map[bs] > 0)
box_center_in_pixel.append([pos_x, pos_y])
box_size = size_pred[bs, :, pos_x, pos_y]  # B, 2, Num
box_yaw_sin_cos = yaw_pred[bs, :, pos_x, pos_y]
box_height = height_pred[bs, :, pos_x, pos_y]
box_offset = offset_pred[bs, :, pos_x, pos_y]
box_velocity = velocity_pred[bs, :, pos_x, pos_y]
pos_x_real = (pos_x + 0.5) * self.grid_size_row - self.extents[0][1]
pos_y_real = (pos_y + 0.5) * self.grid_size_col - self.extents[1][1]
box_center_x = pos_x_real - box_offset[0, :]  # make sure here, pay more attention
box_center_y = pos_y_real - box_offset[1, :]
box_center_z = (box_height[0, :] + box_height[1, :]) / 2.0
box_size_z = box_height[1, :] - box_height[0, :]
box_category = category_pred[bs, :, pos_x, pos_y]
if box_category.shape[1] > 0:
    box_category_num = torch.argmax(box_category, dim=0)
else:
    box_category_num = torch.tensor([], device=box_category.device, dtype=torch.float)
num_bboxes = len(pos_x)
num_box_elem = 10
box_yaw = torch.atan2(box_yaw_sin_cos[0, :], box_yaw_sin_cos[1, :]) * 0.5
bboxes_tensor = torch.zeros(size=(num_bboxes, num_box_elem), dtype=np.float, device=self.device)
bboxes_tensor[:, :] = torch.stack([box_center_x, box_center_y, box_center_z, box_size[0, :],
                                   box_size[1, :], box_size_z, box_yaw, box_velocity[0, :],
                                   box_velocity[1, :], box_category_num], axis=0).transpose(0, 1)
```

``` draw good result
/media/yyg/C14D581BDA18EBFA/code/MotionNet/logs/train_multi_seq/box_512_baseline/epoch_12.pth
```
