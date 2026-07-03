#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


#define ERROR_CHECK(func_name) do {    \
    cudaError_t err; \
    err = cudaGetLastError(); \
    if (cudaSuccess != err) { \
        fprintf(stderr, "CUDA kernel failed : %d in %s in %s at line %d \n", int(err), cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(-1);             \
    }                         \
} while (0)

#define CHECK_CONTIGUOUS(x) do { \
if (!x.is_contiguous()) {        \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n",#x, __FILE__, __LINE__); \
    exit(-1);                    \
    }                            \
} while (0)

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")


#define DEBUG false
#define DEBUG_CUDA false

__global__ void cluster_pixel_kernel(const bool is_training,
                                     const int batch_size,
                                     const float grid_size_row,
                                     const float grid_size_col,
                                     const int blocksPerBatch,
                                     const int GridsPreMemOneBatch,
                                     const float* voxel_count_gt,
                                     const float* blocking_pred,
                                     const float* confidence_pred,
                                     const float* offset_pred,
                                     const float blocking_threshold,
                                     int* traverse_map_before,
                                     int* grid_skip,
                                     int rows,
                                     int cols,
                                     const int grids,
                                     int *traverse_map)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;
    grid_skip += bs_idx * GridsPreMemOneBatch;
    voxel_count_gt += bs_idx * grids;
    blocking_pred += bs_idx * grids;
    confidence_pred += bs_idx * grids;
    offset_pred += bs_idx * grids * 2U;
    traverse_map_before += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;

    grid_skip[grid] = (voxel_count_gt[grid] <= 0.001) ||
            (!is_training && (blocking_pred[grid] < blocking_threshold || confidence_pred[grid] < 0.05));
//    grid_skip[grid] = (voxel_count_gt[grid] <= 0.001) ||
//            (!is_training && (blocking_pred[grid] < 0.05 || confidence_pred[grid] < 0.05));
    // int center_row = offset_pred[grid];
    // int center_col = offset_pred[grid + grids];
    int center_row = floor(offset_pred[grid] / grid_size_row);
    int center_col = floor(offset_pred[grid + grids] / grid_size_col);
    center_row = idx - center_row;
    center_col = idy - center_col;
    center_row = min(max(center_row, 0), rows - 1);
    center_col = min(max(center_col, 0), cols - 1); // to its center
    traverse_map_before[grid] = center_row * cols + center_col; // parent
    traverse_map[grid] = center_row * cols + center_col; // parent
}


__global__ void traverse_node_kernel_part1(const int batch_size,
                                     const int blocksPerBatch,
                                     const int GridsPreMemOneBatch,
                                     int* traverse_map_before,
                                     int* grid_skip,
                                     int rows,
                                     int cols,
                                     const int grids,
                                     int* traverse_map,
                                     int* rank_map)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;

    grid_skip += bs_idx * GridsPreMemOneBatch;
    traverse_map_before += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;
    rank_map += bs_idx * GridsPreMemOneBatch;

    if (grid_skip[grid]) {
        return;
    }
    int curr_grid = grid;
    int num_step = 1;
    while (traverse_map_before[curr_grid] != curr_grid) {
        curr_grid = traverse_map_before[curr_grid];
        ++num_step;
        if (num_step > 50) {
            break;
        }
    }

    atomicMax(rank_map + curr_grid, num_step);
    traverse_map[grid] = curr_grid;
    atomicExch(traverse_map + curr_grid, curr_grid);
}


__device__ void disjoint_set_find(const int* traverse_map, int* grid)
{
    int root_grid = (*grid);
    int num_step = 0;
    while (traverse_map[root_grid] != root_grid) {
        root_grid = traverse_map[root_grid];
        ++num_step;
        if (num_step > 20) {
            printf("[%d, %d] ==> [%d] has more than 20 step\n");
            break;
        }
    }
    *grid = root_grid;
}

__device__ void disjoint_set_union2(int x,
                                    int y,
                                    int *traverse_map,
                                    int *rank_map)
{
    // disjoint_set_find(traverse_map, &x);
    // disjoint_set_find(traverse_map, &y);
    if (x == y) {
        return;
    }
    if (rank_map[x] < rank_map[y]) {
        atomicExch(traverse_map + x, y);
    } else if (rank_map[y] < rank_map[x]) {
        atomicExch(traverse_map + y, x);
    } else {
        if (x < y) {
            // atomicAdd(rank_map + x, 1);
            // atomicExch(traverse_map + y, x);
            atomicExch(traverse_map + y, traverse_map[x]);
        } else {
            // atomicAdd(rank_map + y, 1);
            atomicExch(traverse_map + x, traverse_map[y]);
        }
    }
}


__global__ void disjoint_set_union(const int batch_size,
                                   const int blocksPerBatch,
                                   const int GridsPreMemOneBatch,
                                   int rows,
                                   int cols,
                                   const int grids,
                                   int *traverse_map,
                                   int *rank_map,
                                   int *lock_map)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;

    rank_map += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;

    if (rank_map[grid] < 1) {
        return;
    }
    int min_row = max(idx - 1, 0);
    int max_row = min(idx + 1, rows - 1);
    int min_col = max(idy - 1, 0);
    int max_col = min(idy + 1, cols - 1);

    for (int i = min_row; i <= max_row; ++i) {
        for (int j = min_col; j <= max_col; ++j) {
            int grid2 = i * cols + j;
            if (grid == grid2) {
                continue;
            }
            if ((rank_map[grid] >=1) && (rank_map[grid2] >= 1)) {
                // disjoint_set_union2(grd, grid2, traverse_map, rank_map);
                int x = grid;
                int y = grid2;
                x = traverse_map[x];
                y = traverse_map[y];
                if (x == y) {
                    continue;
                }
                if (rank_map[x] < rank_map[y]) {
                     atomicExch(traverse_map + x, traverse_map[y]);
                } else if (rank_map[y] < rank_map[x]) {
                     atomicExch(traverse_map + y, traverse_map[x]);
                } else {
                    if (x < y) {
                        atomicExch(traverse_map + y, traverse_map[x]);
                    } else {
                         atomicExch(traverse_map + x, traverse_map[y]);
                    }
                }
            }
        }
    }
}

__global__ void build_boxes_result_map_bak(const int batch_size,
                                       const int blocksPerBatch,
                                       const int GridsPreMemOneBatch,
                                       int *grid_skip,
                                       int rows,
                                       int cols,
                                       const int grids,
                                       int *traverse_map,
                                       int *lock_map,
                                       float *boxes_result_map)

{
    /*
     * grid_to_cluster_map: assign pixel id , this pixel belong to which cluster
     * count: cluster num
     *
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;

    grid_skip += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;
    lock_map += bs_idx * GridsPreMemOneBatch;
    boxes_result_map += bs_idx * grids;

    if (grid_skip[grid]) { // skip
        return;
    }
    int root_grid = traverse_map[grid];
    if (atomicAdd(lock_map + root_grid, 1) == 0) {
        atomicExch(boxes_result_map + root_grid, 1.0 * root_grid);
    }
}

__global__ void assign_boxes_result_map_total_points(const int batch_size,
                                                       const int blocksPerBatch,
                                                       const int GridsPreMemOneBatch,
                                                       int *grid_skip,
                                                       int rows,
                                                       int cols,
                                                       const int grids,
                                                       int *traverse_map,
                                                       const int num_view_index_channel,
                                                       const float *view_index,
                                                       const int num_boxes_result_map_channel,
                                                       float *boxes_result_map)

{
    /*
     * grid_to_cluster_map: assign pixel id , this pixel belong to which cluster
     * count: cluster num
     * boxes_result_map: (confidence, vx, vy)
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;
    grid_skip += bs_idx * GridsPreMemOneBatch;
    if (grid_skip[grid]) { // skip
        return;
    }
    traverse_map += bs_idx * GridsPreMemOneBatch;
    view_index += bs_idx * grids * num_view_index_channel;
    boxes_result_map += bs_idx * grids * num_view_index_channel;
    int count_idx = num_view_index_channel * grid;
    const float count = view_index[count_idx];
    int root_grid = traverse_map[grid];
    // root center total points
    atomicAdd(boxes_result_map + root_grid * num_boxes_result_map_channel, count * 1.0);
}

__global__ void build_boxes_result_map(const int batch_size,
                                     const int blocksPerBatch,
                                     const int GridsPreMemOneBatch,
                                     int *grid_skip,
                                     int rows,
                                     int cols,
                                     const int grids,
                                     int *traverse_map,
                                     const int num_view_index_channel,
                                     const float *view_index,
                                     const float *confidence_pred,
                                     const float *velocity_pred,
                                     const int num_boxes_result_map_channel,
                                     float *boxes_result_map)
{
    /*
     * count: cluster num
     * boxes_result_map => (B, H, W, C): (center_points_count, confidence, vx, vy)
     * boxes_result_map: (confidence, vx, vy)
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;
    grid_skip += bs_idx * GridsPreMemOneBatch;
    if (grid_skip[grid]) { // skip
        return;
    }
    traverse_map += bs_idx * GridsPreMemOneBatch;
    view_index += bs_idx * grids * num_view_index_channel;
    boxes_result_map += bs_idx * grids * num_view_index_channel;
    confidence_pred += bs_idx * grids;
    velocity_pred += bs_idx * grids * 2U;
    int count_idx = num_view_index_channel * grid;
    const float count = view_index[count_idx];
    int root_grid = traverse_map[grid];
    // root center total points, box_result_pos also means points number
    float* box_result_pos = boxes_result_map + root_grid * num_boxes_result_map_channel;
    float root_center_total_pts = box_result_pos[0];
    if (root_center_total_pts > 0.001) {
        atomicAdd(box_result_pos + 1U, (confidence_pred[root_grid] * count / root_center_total_pts));
        atomicAdd(box_result_pos + 2U, (velocity_pred[root_grid] * count / root_center_total_pts));
        atomicAdd(box_result_pos + 3U, (velocity_pred[root_grid + grids] * count / root_center_total_pts));
    }
}

__global__ void assign_pixel_id_part(const int batch_size,
                                const int blocksPerBatch,
                                const int GridsPreMemOneBatch,
                                int *grid_skip,
                                int rows,
                                int cols,
                                const int grids,
                                int *traverse_map,
                                int *grid_to_cluster_map,
                                int *lock_map,
                                const float *blocking_gt,
                                int *has_objectness_map,
                                const float *blocking_pred_c,
                                float *total_blocking_prob_map_c,
                                float *total_cluster_count_map_c,
                                const float *view_index,
                                const int num_view_index_channel,
                                float *boxes_result_map)
{
    /*
     * grid_to_cluster_map: assign pixel id , this pixel belong to which cluster
     * count: cluster num
     *
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;

    grid_skip += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;
    has_objectness_map += bs_idx * GridsPreMemOneBatch;
    lock_map += bs_idx * GridsPreMemOneBatch;
    blocking_gt += bs_idx * grids;
    grid_to_cluster_map += bs_idx * GridsPreMemOneBatch;
    boxes_result_map += bs_idx * grids;

    total_blocking_prob_map_c += bs_idx * grids;
    total_cluster_count_map_c += bs_idx * grids;

    blocking_pred_c += bs_idx * grids;

    if (grid_skip[grid]) { // skip
        return;
    }
    view_index += bs_idx * grids * num_view_index_channel;
    int count_idx = num_view_index_channel * grid;
    const float count = view_index[count_idx];
    int root_grid = traverse_map[grid];
    int blocking_gt_val = blocking_gt[grid];
    if (blocking_gt_val > 0.05) { // positive
        atomicExch(has_objectness_map + root_grid, 1);
    }
    atomicAdd(total_blocking_prob_map_c + root_grid, blocking_pred_c[grid]);
    atomicAdd(total_cluster_count_map_c + root_grid, count * 1.0);
    if (atomicAdd(lock_map + root_grid, 1) == 0) { // only one elem can get it root ones
        atomicExch(grid_to_cluster_map + root_grid, root_grid);
        atomicExch(boxes_result_map + root_grid, root_grid);
    }
    atomicExch(grid_to_cluster_map + grid, root_grid);
}

__global__ void build_weights(const int batch_size,
                             const int blocksPerBatch,
                             const int GridsPreMemOneBatch,
                             int *grid_skip,
                             int rows,
                             int cols,
                             int grids,
                             const int *traverse_map,
                             const float *blocking_pred,
                             const float *view_index,
                             const int num_view_index_channel,
                             float *total_blocking_prob_map,
                             float *total_cluster_count_map,
                             int *has_objectness_map,
                             float *blocking_weight,
                             float *confidence_weight,
                             float *velocity_weight)
{
    /*
     * grid_to_cluster_map: assign pixel id , this pixel belong to which cluster
     * count: cluster num
     *
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;
    grid_skip += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;
    has_objectness_map += bs_idx * GridsPreMemOneBatch;
    blocking_weight += bs_idx * grids;
    confidence_weight += bs_idx * grids;
    velocity_weight += bs_idx * grids;
    view_index += bs_idx * grids * num_view_index_channel;
    if (grid_skip[grid]) { // skip, if grid_next_grids
        return;
    }
    total_blocking_prob_map += bs_idx * grids;
    total_cluster_count_map += bs_idx * grids;
    blocking_pred += bs_idx * grids;
    float center_x = 0.0;
    float center_y = 0.0;
    float norm = 0.0;
    int count_idx = num_view_index_channel * grid;
    const float count = view_index[count_idx];
    int root_grid = traverse_map[grid];
    if ((has_objectness_map[root_grid]) > 0 && (count > 0.01)) {
        center_x = view_index[count_idx + 3];
        center_y = view_index[count_idx + 4];
        norm = sqrt(center_x * center_x + center_y * center_y);
        blocking_weight[grid] = 0.001 * norm * count;
        if (total_cluster_count_map[root_grid] > 0.0) {
            float weight = 1.0 / total_cluster_count_map[root_grid];
            velocity_weight[grid] = weight * count;
        }
    }
    // play attention to this, confidence map need not has object ness map more than zero
    if (total_blocking_prob_map[root_grid] > 0.0) {
        float weight = 1.0 / total_blocking_prob_map[root_grid];
        confidence_weight[grid] = weight * blocking_pred[grid] * count;
    }
    // printf("gird [%d] total blocking prob is: %f b_pred %f, count %f, b_weight %f, c_weight %f, norm %f, weight %f\n",
    //       root_grid, total_blocking_prob_map[root_grid], blocking_pred[grid],
    //       count, blocking_weight[grid], confidence_weight[grid], norm,
    //       1.0 / total_blocking_prob_map[root_grid]);
}

__global__ void update_traverse_map(const int batch_size,
                                     const int blocksPerBatch,
                                     const int GridsPreMemOneBatch,
                                     int *grid_skip,
                                     int rows,
                                     int cols,
                                     int *traverse_map)
{
    /*
    * grid_to_cluster_map: assign pixel id , this pixel belong to which cluster
    * count: cluster num
    *
    */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    int bs_idx = blockIdx.y / blocksPerBatch;
    if (idx >= rows || idy >= cols || bs_idx >= batch_size) {
        return;
    }
    int grid = idx * cols + idy;

    grid_skip += bs_idx * GridsPreMemOneBatch;
    traverse_map += bs_idx * GridsPreMemOneBatch;

    if (grid_skip[grid]) { // skip
        return;
    }
    int root_grid = grid;
    int num_step = 0;
    while (traverse_map[root_grid] != root_grid) {
        root_grid = traverse_map[root_grid];
        ++num_step;
        if (num_step > 20) {
//            printf("[%d, %d] ==> [%d] has more than 20 step\n", root_grid / rows, root_grid % cols,
//                root_grid / rows, root_grid % cols);
            break;
        }
    }
    traverse_map[grid] = root_grid;
}

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
                    float *boxes_result_map_c)
{
    if (DEBUG) {
        printf("[INFO] rtree_launcher start\n");
        printf("rows: %d\n", rows);
        printf("cols: %d\n", cols);
        printf("numChannel %d\n", numChannel);
        printf("batch_size %d\n", batch_size);
        printf("blocking_threshold %f\n", blocking_threshold);
        printf("is_training %d\n", is_training);
    }

    int grids = rows * cols;
    int *traverse_map_before,
        *grid_to_cluster_map_c,
        *grid_skip_c,
        *has_objectness_map_c,
        *traverse_map,
        *rank_map,
        *lock_map;

    // must set fixedMem to zeros
    cudaMemset(fixedMem_c, 0, numChannel * grids * sizeof(int) * batch_size);

    traverse_map_before = fixedMem_c;
    grid_to_cluster_map_c = fixedMem_c + grids;
    grid_skip_c = fixedMem_c + 2U * grids;
    has_objectness_map_c = fixedMem_c + 3U * grids;
    traverse_map = fixedMem_c + 4U * grids;
    rank_map = fixedMem_c + 5U * grids;
    lock_map = fixedMem_c + 6U * grids;

    cudaError_t err;
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(((rows + threadsPerBlock.x - 1) / threadsPerBlock.x),
                       ((cols + threadsPerBlock.y - 1) / threadsPerBlock.y) * batch_size);
    int blocksPerBatch = (cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
    int GridsPreMemOneBatch = grids * numChannel;

    if (DEBUG) {
        printf("[INFO] cluster_pixel_kernel\n");
    }

    cluster_pixel_kernel<<<blocksPerGrid, threadsPerBlock>>>(is_training,
                                                             batch_size,
                                                             grid_size_row,
                                                             grid_size_col,
                                                             blocksPerBatch,
                                                             GridsPreMemOneBatch,
                                                             voxel_count_gt_c,
                                                             blocking_pred_c,
                                                             confidence_pred_c,
                                                             offset_pred_c,
                                                             blocking_threshold,
                                                             traverse_map_before,
                                                             grid_skip_c,
                                                             rows,
                                                             cols,
                                                             grids,
                                                             traverse_map);

    ERROR_CHECK("cluster_pixel_kernel");
    if (DEBUG) {
        printf("finished cluster_pixel_kernel.\n");
        printf("start traverse_node_kernel_part1.\n");
    }

    traverse_node_kernel_part1<<<blocksPerGrid, threadsPerBlock>>>(batch_size,
                                                                     blocksPerBatch,
                                                                     GridsPreMemOneBatch,
                                                                     traverse_map_before,
                                                                     grid_skip_c,
                                                                     rows,
                                                                     cols,
                                                                     grids,
                                                                     traverse_map,
                                                                     rank_map);

    ERROR_CHECK("traverse_node_kernel_part1");

    disjoint_set_union<<<blocksPerGrid, threadsPerBlock>>>(
            batch_size, blocksPerBatch, GridsPreMemOneBatch, rows, cols, grids, traverse_map, rank_map, lock_map);

    ERROR_CHECK("disjoint_set_union");

    update_traverse_map<<<blocksPerGrid, threadsPerBlock>>>(batch_size, blocksPerBatch, GridsPreMemOneBatch,
                                                            grid_skip_c, rows, cols, traverse_map);

    ERROR_CHECK("update_traverse_map");

    for (int idx = 0; idx < batch_size; ++idx) {
        cudaMemset(lock_map + idx * GridsPreMemOneBatch, 0, sizeof(int) * grids);
    }

    ERROR_CHECK(" cudaMemset_lock_map");

    assign_boxes_result_map_total_points<<<blocksPerGrid, threadsPerBlock>>>(batch_size,
                                                                             blocksPerBatch,
                                                                             GridsPreMemOneBatch,
                                                                             grid_skip_c,
                                                                             rows,
                                                                             cols,
                                                                             grids,
                                                                             traverse_map,
                                                                             num_view_index_channel,
                                                                             view_index_c,
                                                                             num_boxes_result_map_channel,
                                                                             boxes_result_map_c);

    build_boxes_result_map<<<blocksPerGrid, threadsPerBlock>>>(batch_size,
                                                                 blocksPerBatch,
                                                                 GridsPreMemOneBatch,
                                                                 grid_skip_c,
                                                                 rows,
                                                                 cols,
                                                                 grids,
                                                                 traverse_map,
                                                                 num_view_index_channel,
                                                                 view_index_c,
                                                                 confidence_pred_c,
                                                                 velocity_pred_c,
                                                                 num_boxes_result_map_channel,
                                                                 boxes_result_map_c);

    ERROR_CHECK("build_boxes_result_map");

    if (DEBUG) {
        printf("[INFO] rtree_launcher end\n");
    }
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %d in %s in %s at line %d \n", int(err), cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }
#ifdef DEBUG_CUDA
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif

}


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
                    float *boxes_result_map_c)
{
    if (DEBUG) {
        printf("[INFO] rtree_launcher start\n");
        printf("rows: %d\n", rows);
        printf("cols: %d\n", cols);
        printf("numChannel %d\n", numChannel);
        printf("batch_size %d\n", batch_size);
        printf("blocking_threshold %f\n", blocking_threshold);
        printf("is_training %d\n", is_training);
    }

    cudaError_t err;
    int grids = rows * cols;
    int *traverse_map_before,
        *grid_to_cluster_map_c,
        *grid_skip_c,
        *has_objectness_map_c,
        *traverse_map,
        *rank_map,
        *lock_map;

    float *total_blocking_prob_map_c,
          *total_cluster_count_map_c;
    // must set fixedMem to zeros
    cudaMemset(fixedMem_c, 0, numChannel * grids * sizeof(int) * batch_size);
    cudaMemset(fixedMem_float_c, 0, 2 * grids * sizeof(float) * batch_size);

    traverse_map_before = fixedMem_c;
    grid_to_cluster_map_c = fixedMem_c + grids;
    grid_skip_c = fixedMem_c + 2U * grids;
    has_objectness_map_c = fixedMem_c + 3U * grids;
    traverse_map = fixedMem_c + 4U * grids;
    rank_map = fixedMem_c + 5U * grids;
    lock_map = fixedMem_c + 6U * grids;

    total_blocking_prob_map_c = fixedMem_float_c;
    total_cluster_count_map_c = fixedMem_float_c + grids;

    // init nodes states kernel
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(((rows + threadsPerBlock.x - 1) / threadsPerBlock.x),
                       ((cols + threadsPerBlock.y - 1) / threadsPerBlock.y) * batch_size);
    int blocksPerBatch = (cols + threadsPerBlock.y - 1) / threadsPerBlock.y;
    int GridsPreMemOneBatch = grids * numChannel;
    if (DEBUG) {
        printf("[INFO] cluster_pixel_kernel\n");
    }
    cluster_pixel_kernel<<<blocksPerGrid, threadsPerBlock>>>(is_training,
                                                             batch_size,
                                                             grid_size_row,
                                                             grid_size_col,
                                                             blocksPerBatch,
                                                             GridsPreMemOneBatch,
                                                             voxel_count_gt_c,
                                                             blocking_pred_c,
                                                             confidence_pred_c,
                                                             offset_pred_c,
                                                             blocking_threshold,
                                                             traverse_map_before,
                                                             grid_skip_c,
                                                             rows,
                                                             cols,
                                                             grids,
                                                             traverse_map);
    if (DEBUG) {
        printf("finished cluster_pixel_kernel.\n");
        printf("start traverse_node_kernel_part1.\n");
    }

    traverse_node_kernel_part1<<<blocksPerGrid, threadsPerBlock>>>(batch_size,
                                                                     blocksPerBatch,
                                                                     GridsPreMemOneBatch,
                                                                     traverse_map_before,
                                                                     grid_skip_c,
                                                                     rows,
                                                                     cols,
                                                                     grids,
                                                                     traverse_map,
                                                                     rank_map);

    disjoint_set_union<<<blocksPerGrid, threadsPerBlock>>>(
            batch_size, blocksPerBatch, GridsPreMemOneBatch, rows, cols, grids, traverse_map, rank_map, lock_map);

    update_traverse_map<<<blocksPerGrid, threadsPerBlock>>>(batch_size, blocksPerBatch, GridsPreMemOneBatch,
                                                            grid_skip_c, rows, cols, traverse_map);
    for (int idx = 0; idx < batch_size; ++idx) {
        cudaMemset(lock_map + idx * GridsPreMemOneBatch, 0, sizeof(int) * grids);
    }
    assign_pixel_id_part<<<blocksPerGrid, threadsPerBlock>>>(batch_size,
                                                             blocksPerBatch,
                                                             GridsPreMemOneBatch,
                                                             grid_skip_c,
                                                             rows,
                                                             cols,
                                                             grids,
                                                             traverse_map,
                                                             grid_to_cluster_map_c,
                                                             lock_map,
                                                             blocking_gt_c,
                                                             has_objectness_map_c,
                                                             blocking_pred_c,
                                                             total_blocking_prob_map_c,
                                                             total_cluster_count_map_c,
                                                             view_index_c,
                                                             num_view_index_channel,
                                                             boxes_result_map_c);
    //
    if (DEBUG) {
        printf("CUDA:assign_pixel_id_part finished\n");
    }
    if (is_training) {
        build_weights<<<blocksPerGrid, threadsPerBlock>>>(batch_size,
                                                          blocksPerBatch,
                                                          GridsPreMemOneBatch,
                                                          grid_skip_c,
                                                          rows,
                                                          cols,
                                                          grids,
                                                          traverse_map,
                                                          blocking_pred_c,
                                                          view_index_c,
                                                          num_view_index_channel,
                                                          total_blocking_prob_map_c,
                                                          total_cluster_count_map_c,
                                                          has_objectness_map_c,
                                                          blocking_weight_c,
                                                          confidence_weight_c,
                                                          velocity_weight_c);
        // printf("CUDA:build_weights finished\n");
    }
#ifdef DEBUG_CUDA
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
    if (DEBUG) {
        printf("[INFO] rtree_launcher end\n");
    }
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "[ERROR] CUDA kernel failed : %d in %s in %s at line %d \n", int(err), cudaGetErrorString(err), __FILE__, __LINE__);
        exit(-1);
    }
#ifdef DEBUG_CUDA
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}