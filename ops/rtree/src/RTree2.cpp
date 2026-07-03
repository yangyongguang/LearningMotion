#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>

#define DEBUG false

#define CHECK_CONTIGUOUS(x) do { \
if (!x.is_contiguous()) {        \
    printf("%s must be contiguous tensor at %s:%d\n",#x, __FILE__, __LINE__); \
    exit(-1);                    \
    }                            \
} while (0)


#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")


void rtree_launcher(bool is_training,
                    int rows,
                    int cols,
                    const float grid_size_row,
                    const float grid_size_col,
                    int numChannel,
                    int batch_size,
                    const float blocking_threshold,
                    const float *blocking_pred_c,
                    const float *confidence_pred_c,
                    const float *voxel_count_gt_c,
                    const float *offset_pred_c,
                    const float *blocking_gt_c,
                    const float *view_index_c,
                    const int num_view_index_channel,
                    int *fixedMem_c,
                    float *fixedMem_float_c,
                    float *blocking_weight_c,
                    float *confidence_weight_c,
                    float *velocity_weight_c,
                    float *offset_weight_c,
                    float *boxes_result_map_c);

void rtree_launcher_val(bool is_training,
                    int rows,
                    int cols,
                    const float grid_size_row,
                    const float grid_size_col,
                    int numChannel,
                    int batch_size,
                    const float blocking_threshold,
                    const float *blocking_pred_c,
                    const float *confidence_pred_c,
                    const float *velocity_pred_c,
                    const float *voxel_count_gt_c,
                    const float *offset_pred_c,
                    int *fixedMem_c,
                    const int num_boxes_result_map_channel,
                    const int num_view_index_channel,
                    const float *view_index_c,
                    float *boxes_result_map_c);


int rtree_cuda(bool is_training,
          int rows,
          int cols,
          const float grid_size_row,
          const float grid_size_col,
          float blocking_threshold,
          at::Tensor blocking_pred,
          at::Tensor confidence_pred,
          at::Tensor voxel_count_gt,
          at::Tensor offset_pred,
          at::Tensor blocking_gt,
          at::Tensor view_index,
          int num_view_index_channel,
          int numChannel,
          at::Tensor fixedMem,
          at::Tensor fixedMem_float,
          at::Tensor blocking_weight,
          at::Tensor confidence_weight,
          at::Tensor velocity_weight,
          at::Tensor offset_weight,
          at::Tensor boxes_result_map)
{
    if (DEBUG) {
        printf("[INFO] start rtree\n");
        printf("[INFO] numChannel %d\n", numChannel);
        printf("[INFO] is train %d\n", is_training);
    }

    CHECK_CONTIGUOUS(blocking_pred);
    CHECK_CONTIGUOUS(voxel_count_gt);
    CHECK_CONTIGUOUS(offset_pred);
    CHECK_CONTIGUOUS(blocking_gt);
    CHECK_CONTIGUOUS(fixedMem);
    CHECK_CONTIGUOUS(fixedMem_float);
    CHECK_CONTIGUOUS(offset_weight);
    CHECK_CONTIGUOUS(blocking_weight);
    CHECK_CONTIGUOUS(confidence_weight);
    CHECK_CONTIGUOUS(velocity_weight);

    int batch_size = offset_pred.size(0);
    const float *blocking_pred_data = blocking_pred.data<float>();
    const float *confidence_pred_data = confidence_pred.data<float>();
    const float *voxel_count_gt_data = voxel_count_gt.data<float>();
    const float *offset_pred_data = offset_pred.data<float>();
    const float *blocking_gt_data = blocking_gt.data<float>();
    int *fixedMem_data = fixedMem.data<int>();
    float *fixedMem_float_data = fixedMem_float.data<float>();

    const float *view_index_data = view_index.data<float>();

    float *blocking_weight_data = blocking_weight.data<float>();
    float *confidence_weight_data = confidence_weight.data<float>();
    float *velocity_weight_data = velocity_weight.data<float>();
    float *offset_weight_data = offset_weight.data<float>();

    float *boxes_result_map_data = boxes_result_map.data<float>();

//    rtree_launcher(is_training,
//                   rows,
//                   cols,
//                   grid_size_row,
//                   grid_size_col,
//                   numChannel,
//                   batch_size,
//                   blocking_threshold,
//                   blocking_pred_data,
//                   confidence_pred_data,
//                   voxel_count_gt_data,
//                   offset_pred_data,
//                   blocking_gt_data,
//                   view_index_data,
//                   num_view_index_channel,
//                   fixedMem_data,
//                   fixedMem_float_data,
//                   blocking_weight_data,
//                   confidence_weight_data,
//                   velocity_weight_data,
//                   offset_weight_data,
//                   boxes_result_map_data);

    if (DEBUG) {
        printf("[INFO] end rtree\n");
    }
}


int rtree_cuda_val(bool is_training,
          int rows,
          int cols,
          const float grid_size_row,
          const float grid_size_col,
          float blocking_threshold,
          at::Tensor blocking_pred,
          at::Tensor confidence_pred,
          at::Tensor voxel_count_gt,
          at::Tensor offset_pred,
          at::Tensor velocity_pred,
          const int num_view_index_channel,
          at::Tensor view_index,
          int numChannel,
          at::Tensor fixedMem,
          const int num_boxes_result_map_channel,
          at::Tensor boxes_result_map)
{
    if (DEBUG) {
        printf("[INFO] start rtree val\n");
        printf("[INFO] numChannel %d\n", numChannel);
        printf("[INFO] is train %d\n", is_training);
    }
    CHECK_CONTIGUOUS(blocking_pred);
    CHECK_CONTIGUOUS(voxel_count_gt);
    CHECK_CONTIGUOUS(offset_pred);
    CHECK_CONTIGUOUS(fixedMem);

    int batch_size = offset_pred.size(0);
    const float *blocking_pred_data = blocking_pred.data<float>();
    const float *confidence_pred_data = confidence_pred.data<float>();
    const float *voxel_count_gt_data = voxel_count_gt.data<float>();
    const float *offset_pred_data = offset_pred.data<float>();
    const float *velocity_pred_data = velocity_pred.data<float>();
    const float *view_index_data = view_index.data<float>();
    int *fixedMem_data = fixedMem.data<int>();
    float *boxes_result_map_data = boxes_result_map.data<float>();

    rtree_launcher_val(is_training,
                   rows,
                   cols,
                   grid_size_row,
                   grid_size_col,
                   numChannel,
                   batch_size,
                   blocking_threshold,
                   blocking_pred_data,
                   confidence_pred_data,
                   velocity_pred_data,
                   voxel_count_gt_data,
                   offset_pred_data,
                   fixedMem_data,
                   num_boxes_result_map_channel,
                   num_view_index_channel,
                   view_index_data,
                   boxes_result_map_data);

    if (DEBUG) {
        printf("[INFO] end rtree\n");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rtree_cuda, "rtree function for get pixel cluster center (CUDA)");
    m.def("forward_val", &rtree_cuda_val, "rtree function for get pixel cluster center (CUDA)");
}