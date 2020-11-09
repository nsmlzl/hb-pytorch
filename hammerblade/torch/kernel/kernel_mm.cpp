//====================================================================
// addmm kernel
// 03/09/2020 Kexin Zheng, Lin Cheng (kz73@cornell.edu, lc873@cornell.edu)
//====================================================================

#define BLOCK_DIM 16 // sqrt(4KB/4 byte/4 data matrix) = 15 max
#include <kernel_common.hpp>
#include <kernel_addmm.hpp>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_mm(
          hb_tensor_t* _result,
          hb_tensor_t* _mat1,
          hb_tensor_t* _mat2) {

    auto mat1 = HBTensor<float, 2>(_mat1);
    auto mat2 = HBTensor<float, 2>(_mat2);
    auto result = HBTensor<float, 2>(_result);

    // Start profiling
    bsg_cuda_print_stat_kernel_start();


    // v2: single tile, use blocking
    int r1 = mat1.dim(0);
    int c1 = mat1.dim(1);
    int r2 = mat2.dim(0);
    int c2 = mat2.dim(1);
    //hb_assert(c1 == r2);

    // calculate number of row and col blocks in each matrix
    int m1_num_blk_per_col = (r1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per col
    int m1_num_blk_per_row = (c1 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m1 per row
    int m2_num_blk_per_col = (r2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per col
    int m2_num_blk_per_row = (c2 + BLOCK_DIM - 1) / BLOCK_DIM; // how many blocks in m2 per row

    float sp_mat1[BLOCK_DIM * BLOCK_DIM];
    float sp_mat2[BLOCK_DIM * BLOCK_DIM];
    float sp_result[BLOCK_DIM * BLOCK_DIM];

    for (int rr = __bsg_y; rr < m1_num_blk_per_col; rr += BSG_TILE_GROUP_Y_DIM) {
      for (int rc = __bsg_x; rc < m2_num_blk_per_row; rc += BSG_TILE_GROUP_X_DIM) {

        // initialize scratchpad result (init to 0's)
        reset_sp(sp_result);

        // process mat1 and mat2 for this result block
        // only care about blocks of mat1 in row rr
        // and blocks of mat2 in col rc
        for (int mid = 0; mid < m2_num_blk_per_col; mid++) {
          dram_to_sp_simple_generic(sp_mat1, mat1, rr, mid);
          dram_to_sp_simple_generic(sp_mat2, mat2, mid, rc);
          compute_simple(sp_result, sp_mat1, sp_mat2);
        }

        // copy this block back into DRAM
        sp_to_dram(result, sp_result, rr, rc);
      }
    }
    //   End profiling
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_mm, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*)

}

