//====================================================================
// index add kernel
// 10/22/2021 Niklas Schmelzle (jms854@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <cmath>
#include <limits>
#include <atomic>


extern "C" {
  std::atomic<int> processedIndices __attribute__ ((section (".dram")));


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
          hb_tensor_t* t4_p,
          int32_t* dim_p,
          int32_t* sliceSize_p,
          int32_t* numIndices_p,
          int32_t* dstIdxSize_p) {

    auto dst = HBTensor<float>(t0_p);
    auto src = HBTensor<float>(t1_p);
    auto idx = HBTensor<int32_t>(t2_p);
    auto srcIdxLUT = HBTensor<int32_t>(t3_p);
    auto LUTQueue = HBTensor<int32_t>(t4_p);

    int dim = *dim_p;
    int sliceSize = *sliceSize_p;
    int numIndices = *numIndices_p;
    int dstIdxSize = *dstIdxSize_p;


    // Start profiling
    bsg_cuda_print_stat_kernel_start();
    bsg_saif_start();


    const int IMAX = std::numeric_limits<int>::max();
    if (bsg_id == 0) {
        processedIndices = 0;
    }
    g_barrier.sync();
    int processedIndicesPrev = 0;

    while (processedIndices.load() < numIndices) {
        g_barrier.sync();
        // fill srcIdxLUT for current partial index_add operation
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
                    int queueAddr = processedIndices++;
                    queueAddr -= processedIndicesPrev;
                    LUTQueue(queueAddr) = curDstIdx;
                    break;
                }
                // no match found for curDstIdx -> all were processed
                if (srcIdxIdx+1 == numIndices) {
                    srcIdxLUT(curDstIdx) = IMAX;
                }
            }
        }


        g_barrier.sync();
        int queueSize = processedIndices-processedIndicesPrev;
        processedIndicesPrev = processedIndices;
        // compute partial index_add operation
        for (int linearIndex = bsg_id; linearIndex < queueSize*sliceSize; linearIndex += BSG_TILE_GROUP_X_DIM*BSG_TILE_GROUP_Y_DIM) {
            int dstIndex = LUTQueue(linearIndex / sliceSize);
            int srcIndex = srcIdxLUT(dstIndex);
            int elementInSlice = linearIndex % sliceSize;

            int dst_element_idx = get_element_index(dst, dim, dstIndex, elementInSlice);
            int src_element_idx = get_element_index(src, dim, srcIndex, elementInSlice);

            dst(dst_element_idx) += src(src_element_idx);
        }
    }


    g_barrier.sync();
    // End profiling
    bsg_saif_end();
    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_index_add, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_tensor_t*, hb_tensor_t*, int32_t*, int32_t*, int32_t*, int32_t*)

}

