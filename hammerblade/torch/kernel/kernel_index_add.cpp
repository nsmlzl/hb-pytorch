//====================================================================
// index add kernel
// 10/22/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>
#include "bsg_manycore_atomic.h"

// equal to number of tiles
#define LOCK_SIZE 128


extern "C" {
  int lock[LOCK_SIZE] __attribute__ ((section (".dram")));

  void init_locks() {
    for (int linearIndex = bsg_id; linearIndex < LOCK_SIZE; linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
        lock[linearIndex] = 0;
    }
  }

  void aquire_lock(int *lock) {
      int lock_ret = 1;
      do {
          lock_ret = bsg_amoswap_aq(lock, 1);
      } while (lock_ret != 0);
  }

  void release_lock(int *lock) {
      bsg_amoswap_rl(lock, 0);
  }


  int get_element_index(HBTensor<float> &ten, int add_dim, int index, int elementInSlice) {
      int offset = 0;
      for (int i = ten.ndim()-1; i > 0; --i) {
          int size = (i == add_dim)? 1 : ten.dim(i);
          offset += (elementInSlice % size) * ten.stride(i);
          elementInSlice /= size;
      }
      offset += elementInSlice * ten.stride(0) + index * ten.stride(add_dim);
      if (offset > ten.numel()) {
          bsg_printf("Warning: index out of range!\n");
          offset = 0;
      }
      return offset;
  }

  __attribute__ ((noinline))  int tensorlib_index_add_small_index(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          int32_t* dim_p,
          int32_t* sliceSize_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int32_t>(t2_p);
    int dim = *dim_p;
    int sliceSize = *sliceSize_p;

    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    init_locks();
    g_barrier.sync();

    for (int srcIndex = 0; srcIndex < idx.numel(); ++srcIndex) {
        int dstIndex = idx(srcIndex);
        for (int linearIndex = bsg_id; linearIndex < sliceSize; linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
            int dst_element_idx = get_element_index(dst, dim, dstIndex, linearIndex);
            int src_element_idx = get_element_index(src, dim, srcIndex, linearIndex);

            int dst_lock_idx = dst_element_idx && 0xFF;
            int *dst_lock = &lock[dst_lock_idx];

            aquire_lock(dst_lock);
            dst(dst_element_idx) += src(src_element_idx);
            release_lock(dst_lock);
        }
    }

    //   End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }


  __attribute__ ((noinline))  int tensorlib_index_add_large_index(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          int32_t* dim_p,
          int32_t* sliceSize_p,
          int32_t* numIndices_p,
          int32_t* indexMajorMode_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int32_t>(t2_p);
    int dim = *dim_p;
    int sliceSize = *sliceSize_p;
    int numIndices = *numIndices_p;
    int indexMajorMode = *indexMajorMode_p;

      // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    int idxPtrSize = dst.dim(dim);
    int *idxPtr = (int*)malloc(idxPtrSize * sizeof(int));
    int partition = 0;
    int elementCnt = 0;
    // TODO: reuse previous idxPtr
    while (elementCnt < numIndices) { // && bsg_id == 0) {
        // init idxPtr
        for (int i = 0; i < idxPtrSize; i++) {
            idxPtr[i] = -2;
        }
        // set idxPtr for current partition
        // TODO: Complexity quadatic -> bad!
        for (int cur_idx = 0; cur_idx < idxPtrSize; cur_idx++) {
            int count = 0;
            for (int i = 0; i < numIndices; i++) {
                if (cur_idx == idx(i) && count == partition) {
                    //bsg_printf("%d: found match\n", cur_idx);
                    idxPtr[cur_idx] = i;
                    ++elementCnt;
                    //idxPtrEmpty = false;
                    i = idxPtrSize;
                    break;
                } else if (cur_idx == idx(i) && count != partition) {
                    //bsg_printf("%d: found match; but wrong partition\n", cur_idx);
                    ++count;
                }
                if (i+1 == numIndices) {
                    //bsg_printf("%d: no match found\n", cur_idx);
                    idxPtr[cur_idx] = -1;
                }
            }
        }

        if (bsg_id == 0) {
            for (int i = 0; i < idxPtrSize; i++) {
                hb_assert_msg(idxPtr[i] >= -1, "error: each value in idxPtr array should be greater than -1 (value %d)\n", idxPtr[i]);
                //bsg_printf("idx[%d]=%d\n", i, idxPtr[i]);
            }
        }

        for (int linearIndex = bsg_id; linearIndex < idxPtrSize*sliceSize; linearIndex += BSG_TILE_GROUP_X_DIM * BSG_TILE_GROUP_Y_DIM) {
            int dstIndex = linearIndex / sliceSize;
            int srcIndex = idxPtr[dstIndex];
            if (srcIndex != -1) {
                int elementInSlice = linearIndex % sliceSize;

                //bsg_printf("tile %d, srcIndex: %d, elementInSlice: %d\n", bsg_id, srcIndex, elementInSlice);

                int dst_element_idx = get_element_index(dst, dim, dstIndex, elementInSlice);
                int src_element_idx = get_element_index(src, dim, srcIndex, elementInSlice);

                dst(dst_element_idx) += src(src_element_idx);
            }
        }


        ++partition;
        g_barrier.sync();
    }


    //   End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_index_add_small_index, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     int32_t*, int32_t*)
  HB_EMUL_REG_KERNEL(tensorlib_index_add_large_index, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     int32_t*, int32_t*, int32_t*, int32_t*)

}

