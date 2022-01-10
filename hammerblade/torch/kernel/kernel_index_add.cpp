//====================================================================
// index add kernel
// 10/22/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>
#include <limits>
#include <atomic>

// TODO: why unable to run atomic<int> ?
//std::atomic<int> processedIndices;
volatile int processedIndices = 42;


extern "C" {
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


  __attribute__ ((noinline))  int tensorlib_index_add(
          hb_tensor_t* t0_p,
          hb_tensor_t* t1_p,
          hb_tensor_t* t2_p,
          hb_tensor_t* t3_p,
          int32_t* dim_p,
          int32_t* sliceSize_p,
          int32_t* numIndices_p,
          int32_t* dstIdxSize_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int32_t>(t2_p);
    auto srcIdxLUT = HBTensor<int32_t>(t3_p);

    int dim = *dim_p;
    int sliceSize = *sliceSize_p;
    int numIndices = *numIndices_p;
    int dstIdxSize = *dstIdxSize_p;


    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();

    const int IMAX = std::numeric_limits<int>::max();
    // TODO: remove bsg_printf for profiling

    // TODO: move to host device
    for (int i = bsg_id; i < dstIdxSize; i += BSG_TILE_GROUP_X_DIM*BSG_TILE_GROUP_Y_DIM) {
        srcIdxLUT(i) = -1;
    }
    if (bsg_id == 0) {
        processedIndices = 0;
    }
    g_barrier.sync();

    /*
    if (bsg_id == 0) {
        bsg_printf("numel=%d\n", srcIdxLUT.numel());
        for (int i = 0; i < dstIdxSize; i++) {
            bsg_printf("srcIdxLUT[%d]=%d\n", i, srcIdxLUT(i));
        }
        int tmp = processedIndices;
        bsg_printf("processedIndices=%d\n", tmp);
    }
    if (bsg_id == 1) {
        int tmp = processedIndices;
        bsg_printf("1: processedIndices=%d\n", tmp);
    }


    if (bsg_id == 0) bsg_printf("init done!\n");
    g_barrier.sync();
    */

    bool finished = false;
    while (!finished) {
    //while (processedIndices < numIndices) {
        // fill srcIdxLUT for current computation stage
        for (int curDstIdx = bsg_id; curDstIdx < dstIdxSize; curDstIdx += BSG_TILE_GROUP_X_DIM*BSG_TILE_GROUP_Y_DIM) {
            int srcIdxIdx = srcIdxLUT(curDstIdx);
            // last partition already found highest possible index
            if (srcIdxIdx == numIndices-1) {
                srcIdxLUT(curDstIdx) = IMAX;
            }
            while (srcIdxIdx < numIndices-1) {
                srcIdxIdx++;
                // found new Index
                if (curDstIdx == ((int)idx(srcIdxIdx))) {
                    srcIdxLUT(curDstIdx) = srcIdxIdx;
                    //processedIndices++;
                    //if (bsg_id == 0) bsg_printf("found\n");
                    break;
                }
                // no match found for curDstIdx -> all were processed
                if (srcIdxIdx+1 == numIndices) {
                    //if (bsg_id == 0) bsg_printf("don't found\n");
                    srcIdxLUT(curDstIdx) = IMAX;
                }
            }
        }
        g_barrier.sync();
        // TODO: replace with processedIndices counter
        // Check if srcIdxLUT empty - all elements processed
        finished = true;
        for (int i = 0; i < dstIdxSize; i++) {
            if (srcIdxLUT(i) != IMAX) {
                finished = false;
                break;
            }
        }
        g_barrier.sync();
        if (finished) {
            //if (bsg_id == 0) bsg_printf("stopping, since finished.\n");
            break;
        }

        /*
        if (bsg_id == 0) {
            bsg_printf("srcIdxLUT written.\n");
            bsg_printf("bsg_id: %d\n", bsg_id);
            //int tmp = processedIndices;
            //bsg_printf("processedIndices: %d\n", tmp);
            for(int i = 0; i < dstIdxSize; i++) {
                bsg_printf("srcIdxLUT[%d]=%d\n", i, srcIdxLUT(i));
            }
            for(int i = 0; i < 10; i++) {
                bsg_printf("idx[%d]=%d\n", i, idx(i));
            }
            if (finished) bsg_printf("processing finished\n");
            else bsg_printf("processing not finished\n");
        }
        */


        for (int linearIndex = bsg_id; linearIndex < dstIdxSize*sliceSize; linearIndex += BSG_TILE_GROUP_X_DIM*BSG_TILE_GROUP_Y_DIM) {
            int dstIndex = linearIndex / sliceSize;
            int srcIndex = srcIdxLUT(dstIndex);
            if (srcIndex < numIndices) {
                int elementInSlice = linearIndex % sliceSize;

                int dst_element_idx = get_element_index(dst, dim, dstIndex, elementInSlice);
                int src_element_idx = get_element_index(src, dim, srcIndex, elementInSlice);

                dst(dst_element_idx) += src(src_element_idx);
            }
        }

        g_barrier.sync();
        //if (bsg_id == 0) bsg_printf("partition processed.\n");
    }


    g_barrier.sync();
    // End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_index_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, int32_t*, int32_t*, int32_t*, int32_t*)

}

