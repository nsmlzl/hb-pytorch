//====================================================================
// Experimental block level systolic array for GEMM
// 08/13/2020 Lin Cheng
//====================================================================

#include <kernel_common.hpp>

// Eyeriss buffer setup
#define FILTER_BUF_SIZE  16 // 5 * 3 = 15 -> 16
#define   IMAP_BUF_SIZE  64 // 32 *2 = 64
#define   PSUM_BUF_SIZE 168 // 28 * 3 * 2 = 168
#define       LOAD_PSUM   0

// Eyeriss config
// we use filter-use scheme -- filter stays constant within a process pass
#define IMAGES_PER_BURST 2
#define FILTERS_PER_PROCESSING_PASS 3

template <size_t TRANS_SIZE>
inline void spm_cpy(float* dst, float* src) {
  // compile time branch
  if (TRANS_SIZE % 8 == 0) {
    for (size_t i = 0; i < TRANS_SIZE; i += 8) {
      register float tmp0 = *(src + 0);
      register float tmp1 = *(src + 1);
      register float tmp2 = *(src + 2);
      register float tmp3 = *(src + 3);
      register float tmp4 = *(src + 4);
      register float tmp5 = *(src + 5);
      register float tmp6 = *(src + 6);
      register float tmp7 = *(src + 7);
      asm volatile("": : :"memory");
      *(dst + 0) = tmp0;
      *(dst + 1) = tmp1;
      *(dst + 2) = tmp2;
      *(dst + 3) = tmp3;
      *(dst + 4) = tmp4;
      *(dst + 5) = tmp5;
      *(dst + 6) = tmp6;
      *(dst + 7) = tmp7;
      src += 8;
      dst += 8;
    }
  } else {
    size_t i = 0;
    for (;i < TRANS_SIZE - 7; i += 8) {
      register float tmp0 = *(src + 0);
      register float tmp1 = *(src + 1);
      register float tmp2 = *(src + 2);
      register float tmp3 = *(src + 3);
      register float tmp4 = *(src + 4);
      register float tmp5 = *(src + 5);
      register float tmp6 = *(src + 6);
      register float tmp7 = *(src + 7);
      asm volatile("": : :"memory");
      *(dst + 0) = tmp0;
      *(dst + 1) = tmp1;
      *(dst + 2) = tmp2;
      *(dst + 3) = tmp3;
      *(dst + 4) = tmp4;
      *(dst + 5) = tmp5;
      *(dst + 6) = tmp6;
      *(dst + 7) = tmp7;
      src += 8;
      dst += 8;
    }
    for (;i < TRANS_SIZE; i++) {
      *dst = *src;
      dst++;
      src++;
    }
  }
}

extern "C" {

  __attribute__ ((noinline))  int tensorlib_eyeriss(
    hb_tensor_t* output,
    hb_tensor_t* input,
    hb_tensor_t* weight,
    hb_vector_t* padding,
    hb_vector_t* strides) {


    HBTensor<float> omap(output);
    HBTensor<float> imap(input);
    HBTensor<float> filter(weight);

    // config
    // 0 -- idle       -- do nothing
    // 1 -- filter DMA -- push to 2 to the East
    // 2 -- imap DMA   -- push to NE
    // 3 -- psum DMA   -- push to 2 to the North
    // 4 -- compute    -- push to NE & N

    // char eyeriss_5x14_lenet[8][16] = {
    //     {0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
    //     {1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    //     {1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    //     {1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    //     {1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    //     {1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    //     {0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0},
    //     {0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
    // };

    char eyeriss_5x6x2_lenet[8][16] = {
        {0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0},
        {0, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0},
        {3, 2, 4, 4, 4, 4, 4, 5, 3, 2, 4, 4, 4, 4, 4, 5},
        {3, 2, 4, 4, 4, 4, 4, 5, 3, 2, 4, 4, 4, 4, 4, 5},
        {3, 2, 4, 4, 4, 4, 4, 5, 3, 2, 4, 4, 4, 4, 4, 5},
        {3, 2, 4, 4, 4, 4, 4, 5, 3, 2, 4, 4, 4, 4, 4, 5},
        {3, 2, 4, 4, 4, 4, 4, 5, 3, 2, 4, 4, 4, 4, 4, 5},
        {3, 0, 4, 4, 4, 4, 4, 5, 3, 0, 4, 4, 4, 4, 4, 5},
    };

    // appendix -- defines what should you do
    // 0000     -- normal passing
    // 0001     -- do not pass fliter
    // 0010     -- do not pass imap
    // 0100     -- psum is 2 to the South
    // 1000     -- filter is 2 to the East
    //
    // for DMAs, the appendix defines the row idx

    // char eyeriss_5x14_lenet_appendix[8][16] = {
    //     {0, 0, 0,   1, 2, 3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13},
    //     {0, 0, 0xa, 2, 2, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2,  3},
    //     {1, 0, 8,   0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  3},
    //     {2, 1, 8,   0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  3},
    //     {3, 2, 8,   0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  3},
    //     {4, 3, 0xc, 4, 4, 4, 4,  4,  4,  4,  4,  4,  4,  4,  4,  7},
    //     {0, 4, 5,   6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  0},
    //     {0, 0, 0,   0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    // };

    char eyeriss_5x6x2_lenet_appendix[8][16] = {
        {0, 0, 4,   3, 2, 1, 0,   0, 0,  0, 4,   3, 2, 1, 0,   0},
        {0, 4, 3,   2, 1, 0, 0,   0, 0, 10, 9,   8, 7, 6, 0,   0},
        {0, 5, 0xc, 8, 8, 8, 0xa, 0, 0, 11, 0xc, 8, 8, 8, 0xa, 6},
        {0, 6, 4,   0, 0, 0, 2,   1, 0, 12, 4,   0, 0, 0, 2,   7},
        {0, 7, 4,   0, 0, 0, 2,   2, 0, 13, 4,   0, 0, 0, 2,   8},
        {0, 8, 4,   0, 0, 0, 2,   3, 0, 14, 4,   0, 0, 0, 2,   9},
        {0, 9, 4,   0, 0, 0, 2,   4, 0, 15, 4,   0, 0, 0, 2,   10},
        {0, 0, 7,   3, 3, 3, 3,   5, 0,  0, 7,   3, 3, 3, 3,   11},
    };

    // active config
    char (&mc_config)[8][16] = eyeriss_5x6x2_lenet;
    char (&mc_append)[8][16] = eyeriss_5x6x2_lenet_appendix;
    char tile_config = mc_config[bsg_y][bsg_x];
    char tile_append = mc_append[bsg_y][bsg_x];

    // Eyeriss buffers
    //
    //   imap[#images]     [#in_channel] [row][col]
    //   omap[#images]     [#out_channel][row][col]
    // filter[#out_channel][#in_channel] [ROW][COL]

    float filter_buf_A[FILTER_BUF_SIZE];
    float   imap_buf_A[IMAP_BUF_SIZE];
    float   psum_buf_A[PSUM_BUF_SIZE];

    float filter_buf_B[FILTER_BUF_SIZE];
    float   imap_buf_B[IMAP_BUF_SIZE];
    float   psum_buf_B[PSUM_BUF_SIZE];

    float *filter_buf = filter_buf_A;
    float   *imap_buf =   imap_buf_A;
    float   *psum_buf =   psum_buf_A;

    float *filter_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+1,filter_buf_A)); // East
    float   *imap_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y+1,imap_buf_A)); // NorthEast
    float   *psum_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y,psum_buf_A));   // North

    float *filter_buf_B_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+1,filter_buf_B)); // East
    float   *imap_buf_B_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y+1,imap_buf_B)); // NorthEast
    float   *psum_buf_B_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y,psum_buf_B));   // North

    // filter DMA
    if (tile_config == 1) {
      filter_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+2,filter_buf_A)); // East x 2
      filter_buf_B_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+2,filter_buf_B)); // East x 2
    }

    // psum DMA
    if (tile_config == 3) {
      psum_buf_A_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+2,bsg_y,psum_buf_A));   // North x 2
      psum_buf_B_remote = reinterpret_cast<float*>(bsg_tile_group_remote_pointer(bsg_x+2,bsg_y,psum_buf_B));   // North x 2
    }

    float *filter_buf_remote = filter_buf_A_remote;
    float   *imap_buf_remote =   imap_buf_A_remote;
    float   *psum_buf_remote =   psum_buf_A_remote;

    // sync flags
    // 0 -> ready to load
    // 1 -> ready to use

    volatile unsigned int  filter_A_f      = 0;
    volatile unsigned int  filter_A_f_E    = 0;
    volatile unsigned int *filter_A_f_E_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+1,&filter_A_f));
    volatile unsigned int *filter_A_f_W_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-1,&filter_A_f_E));

    volatile unsigned int  psum_A_f        = 0;
    volatile unsigned int  psum_A_f_N      = 0;
    volatile unsigned int *psum_A_f_N_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y,&psum_A_f));
    volatile unsigned int *psum_A_f_S_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-1,bsg_y,&psum_A_f_N));

    volatile unsigned int  imap_A_f        = 0;
    volatile unsigned int  imap_A_f_NE     = 0;
    volatile unsigned int *imap_A_f_NE_r   = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y+1,&imap_A_f));
    volatile unsigned int *imap_A_f_SW_r   = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-1,bsg_y-1,&imap_A_f_NE));

    volatile unsigned int  filter_B_f      = 0;
    volatile unsigned int  filter_B_f_E    = 0;
    volatile unsigned int *filter_B_f_E_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+1,&filter_B_f));
    volatile unsigned int *filter_B_f_W_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-1,&filter_B_f_E));

    volatile unsigned int  psum_B_f        = 0;
    volatile unsigned int  psum_B_f_N      = 0;
    volatile unsigned int *psum_B_f_N_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y,&psum_B_f));
    volatile unsigned int *psum_B_f_S_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-1,bsg_y,&psum_B_f_N));

    volatile unsigned int  imap_B_f        = 0;
    volatile unsigned int  imap_B_f_NE     = 0;
    volatile unsigned int *imap_B_f_NE_r   = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+1,bsg_y+1,&imap_B_f));
    volatile unsigned int *imap_B_f_SW_r   = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-1,bsg_y-1,&imap_B_f_NE));

    volatile unsigned int  psum_C_f        = 1;
    volatile unsigned int  psum_C_f_N      = 0;
    volatile unsigned int  psum_C_f_N_r    = 0;
    volatile unsigned int  psum_C_f_S_r    = 0;

    volatile unsigned int  imap_C_f        = 1;
    volatile unsigned int  imap_C_f_NE     = 0;
    volatile unsigned int  imap_C_f_NE_r   = 0;
    volatile unsigned int  imap_C_f_SW_r   = 0;

    // filter DMA
    if (tile_config == 1) {
      filter_A_f_E_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+2,&filter_A_f)); // East x 2
      filter_A_f_W_r = NULL;
      filter_B_f_E_r = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y+2,&filter_B_f)); // East x 2
      filter_B_f_W_r = NULL;
    }

    // psum DMA
    if (tile_config == 3) {
      psum_A_f_N_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+2,bsg_y,&psum_A_f));  // North x 2
      psum_A_f_S_r = NULL;
      psum_B_f_N_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x+2,bsg_y,&psum_B_f));  // North x 2
      psum_B_f_S_r = NULL;
    }

    // first col of PE
    if (tile_config == 4 && (tile_append & 8)) {
      filter_A_f_W_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-2,&filter_A_f_E)); // West x 2
      filter_B_f_W_r  = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x,bsg_y-2,&filter_B_f_E)); // West x 2
    }

    // bottom row of PE
    if (tile_config == 4 && (tile_append & 4)) {
      psum_A_f_S_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-2,bsg_y,&psum_A_f_N));  // South x 2
      psum_B_f_S_r    = reinterpret_cast<volatile unsigned int*>(bsg_tile_group_remote_pointer(bsg_x-2,bsg_y,&psum_B_f_N));  // South x 2
    }

    // proxy flags for supporting double buffering

    volatile unsigned int *filter_f        = &filter_A_f;
    volatile unsigned int *filter_f_E      = &filter_A_f_E;
    volatile unsigned int *filter_f_E_r    = filter_A_f_E_r;
    volatile unsigned int *filter_f_W_r    = filter_A_f_W_r;

    volatile unsigned int *psum_f          = &psum_A_f;
    volatile unsigned int *psum_f_N        = &psum_A_f_N;
    volatile unsigned int *psum_f_N_r      = psum_A_f_N_r;
    volatile unsigned int *psum_f_S_r      = psum_A_f_S_r;

    volatile unsigned int *imap_f          = &imap_A_f;
    volatile unsigned int *imap_f_NE       = &imap_A_f_NE;
    volatile unsigned int *imap_f_NE_r     = imap_A_f_NE_r;
    volatile unsigned int *imap_f_SW_r     = imap_A_f_SW_r;

    bool buffer_A = true;

    // Conv2d parameters
    auto N    = omap.dim(0); // number of minibatches
    auto Cout = omap.dim(1); // number of output channels
    auto Hout = omap.dim(2);
    auto Wout = omap.dim(3);
    auto Cin  = imap.dim(1); // number of input channels
    auto Hin  = imap.dim(2);
    auto Win  = imap.dim(3);
    auto Hk   = filter.dim(2);
    auto Wk   = filter.dim(3);

    // functors
    auto filterDMA = [&]() {

      float* src_base = (float*)filter.data_ptr();
      uint32_t* src_strides = filter.get_strides();
      // XXX: hacky -- there is only one channel -- always == 0
      src_base += 0 * src_strides[1] + (uint32_t)tile_append * src_strides[2];

      for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {

        size_t buf_offset = 0;
        // wait until remote filter buffer is ready
        bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (filter_f_E)), 0);
        for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
          // for (size_t col = 0; col < Wk; col++) {
          //   filter_buf_remote[buf_offset] = *(src_base + col); // src_strides[3] has to be 1
          //   buf_offset++;
          // }
          // Unroll -- here we know Wk == 5
          register float filter_w_0 = *(src_base + 0);
          register float filter_w_1 = *(src_base + 1);
          register float filter_w_2 = *(src_base + 2);
          register float filter_w_3 = *(src_base + 3);
          register float filter_w_4 = *(src_base + 4);
          asm volatile("": : :"memory");
          filter_buf_remote[buf_offset + 0] = filter_w_0;
          filter_buf_remote[buf_offset + 1] = filter_w_1;
          filter_buf_remote[buf_offset + 2] = filter_w_2;
          filter_buf_remote[buf_offset + 3] = filter_w_3;
          filter_buf_remote[buf_offset + 4] = filter_w_4;
          buf_offset += 5;
          src_base += src_strides[0];
        }
        asm volatile("": : :"memory");
        *filter_f_E = 1;
        *filter_f_E_r = 1;

        // switch buffer
        if (filter_buf_remote == filter_buf_A_remote) {
          filter_buf_remote = filter_buf_B_remote;
          filter_f_E        = &filter_B_f_E;
          filter_f_E_r      = filter_B_f_E_r;
        } else {
          filter_buf_remote = filter_buf_A_remote;
          filter_f_E        = &filter_A_f_E;
          filter_f_E_r      = filter_A_f_E_r;
        }
      }
    };

    auto imapDMA = [&]() {

      float* src_base = (float*)imap.data_ptr();
      uint32_t* src_strides = imap.get_strides();
      // XXX: hacky -- there is only one channel -- always == 0
      src_base += 0 * src_strides[1] + (uint32_t)tile_append * src_strides[2];

      for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {

        float* src_ptr = src_base;

        for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {

          size_t buf_offset = 0;

          // wait until remote imap buffer is ready
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (imap_f_NE)), 0);
          for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
            for (size_t col = 0; col < Win; col += 8) {
              // imap_buf_remote[buf_offset] = imap(image_id+images,0,(bsg_x-1)+(bsg_y-1),col);
              register float tmp0 = *(src_ptr + col + 0);
              register float tmp1 = *(src_ptr + col + 1);
              register float tmp2 = *(src_ptr + col + 2);
              register float tmp3 = *(src_ptr + col + 3);
              register float tmp4 = *(src_ptr + col + 4);
              register float tmp5 = *(src_ptr + col + 5);
              register float tmp6 = *(src_ptr + col + 6);
              register float tmp7 = *(src_ptr + col + 7);
              asm volatile("": : :"memory");
              imap_buf_remote[buf_offset + 0] = tmp0;
              imap_buf_remote[buf_offset + 1] = tmp1;
              imap_buf_remote[buf_offset + 2] = tmp2;
              imap_buf_remote[buf_offset + 3] = tmp3;
              imap_buf_remote[buf_offset + 4] = tmp4;
              imap_buf_remote[buf_offset + 5] = tmp5;
              imap_buf_remote[buf_offset + 6] = tmp6;
              imap_buf_remote[buf_offset + 7] = tmp7;
              buf_offset += 8;
            }
            src_ptr += src_strides[0];
          }
          asm volatile("": : :"memory");
          *imap_f_NE = 1;
          *imap_f_NE_r = 1;

          // switch buffer
          if (imap_buf_remote == imap_buf_A_remote) {
            imap_buf_remote = imap_buf_B_remote;
            imap_f_NE       = &imap_B_f_NE;
            imap_f_NE_r     = imap_B_f_NE_r;
          } else {
            imap_buf_remote = imap_buf_A_remote;
            imap_f_NE       = &imap_A_f_NE;
            imap_f_NE_r     = imap_A_f_NE_r;
          }
        }
      }
    };

    auto psumDMA = [&]() {
      for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
        for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {
          size_t buf_offset = 0;

          // wait until remote psum buffer is ready
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f_N)), 0);

          // brand new psum
          if (!LOAD_PSUM) {
            // All buffers need to be a multiply of 8
            for (size_t offset = 0; offset < PSUM_BUF_SIZE; offset += 8) {
              psum_buf_remote[offset + 0] = 0;
              psum_buf_remote[offset + 1] = 0;
              psum_buf_remote[offset + 2] = 0;
              psum_buf_remote[offset + 3] = 0;
              psum_buf_remote[offset + 4] = 0;
              psum_buf_remote[offset + 5] = 0;
              psum_buf_remote[offset + 6] = 0;
              psum_buf_remote[offset + 7] = 0;
            }
          } else {
            // XXX: not used in LeNet-5 conv-1
            // read from previous psum
            /*
            for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
              for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
                for (size_t col = 0; col < Wout; col++) {
                  psum_buf_remote[buf_offset] = omap(image_id+images,filter_id+filters,bsg_x-2,col);
                  buf_offset++;
                }
              }
            }
            */
          }
          asm volatile("": : :"memory");
          *psum_f_N = 1;
          *psum_f_N_r = 1;

          // switch buffer
          if (psum_buf_remote == psum_buf_A_remote) {
            psum_buf_remote = psum_buf_B_remote;
            psum_f_N = &psum_B_f_N;
            psum_f_N_r = psum_B_f_N_r;
          } else {
            psum_buf_remote = psum_buf_A_remote;
            psum_f_N = &psum_A_f_N;
            psum_f_N_r = psum_A_f_N_r;
          }
        }
      }
    };

    auto psumWBDMA = [&]() {

      float* dest_base = (float*)omap.data_ptr();
      uint32_t* dest_strides = omap.get_strides();
      dest_base += (uint32_t)tile_append * dest_strides[2];

      for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {
        float* dest_pass = dest_base + filters * dest_strides[1];
        for (size_t images = 0; images < N; images += IMAGES_PER_BURST) {
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f)), 1);
          // write back to omap
          size_t buf_offset = 0;
          for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
            float* dest_ptr = dest_pass;
            for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
              // XXX: this unroll by 4 decision is made with input knowledge
              for (size_t col = 0; col < Wout; col += 4) {
                register float tmp0 = psum_buf[buf_offset + 0];
                register float tmp1 = psum_buf[buf_offset + 1];
                register float tmp2 = psum_buf[buf_offset + 2];
                register float tmp3 = psum_buf[buf_offset + 3];
                asm volatile("": : :"memory");
                *(dest_ptr + col + 0) = tmp0;
                *(dest_ptr + col + 1) = tmp1;
                *(dest_ptr + col + 2) = tmp2;
                *(dest_ptr + col + 3) = tmp3;
                buf_offset += 4;
              }
              dest_ptr += dest_strides[1];
            }
            dest_pass += dest_strides[0];
          }
          // signal psum and imap free
          asm volatile("": : :"memory");
          *psum_f = 0;
          *psum_f_S_r = 0;

          // switch buffer
          if (psum_buf == psum_buf_A) {
            psum_buf        = psum_buf_B;
            psum_f          = &psum_B_f;
            psum_f_S_r      = psum_B_f_S_r;
          } else {
            psum_buf        = psum_buf_A;
            psum_f          = &psum_A_f;
            psum_f_S_r      = psum_A_f_S_r;
          }
        }
      }
    };

    auto computePE = [&]() {

      buffer_A = false;

      psum_f          = &psum_C_f;
      psum_f_N        = &psum_C_f_N;
      psum_f_N_r      = &psum_C_f_N_r;
      psum_f_S_r      = &psum_C_f_S_r;

      imap_f          = &imap_C_f;
      imap_f_NE       = &imap_C_f_NE;
      imap_f_NE_r     = &imap_C_f_NE_r;
      imap_f_SW_r     = &imap_C_f_SW_r;

      imap_buf_remote =   imap_buf_B_remote;
      psum_buf_remote =   psum_buf_B_remote;

      imap_buf =   imap_buf_B;
      psum_buf =   psum_buf_B;

      for (size_t filters = 0; filters < Cout; filters += FILTERS_PER_PROCESSING_PASS) {

        // wait until filter buf is filled
        bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (filter_f)), 1);

        // load filters into registers
        // we are doing 3 filters per pass, and the filter has 5 weights each
        register float filter_w_0_0 = filter_buf[0];
        register float filter_w_0_1 = filter_buf[1];
        register float filter_w_0_2 = filter_buf[2];
        register float filter_w_0_3 = filter_buf[3];
        register float filter_w_0_4 = filter_buf[4];
        register float filter_w_1_0 = filter_buf[5];
        register float filter_w_1_1 = filter_buf[6];
        register float filter_w_1_2 = filter_buf[7];
        register float filter_w_1_3 = filter_buf[8];
        register float filter_w_1_4 = filter_buf[9];
        register float filter_w_2_0 = filter_buf[10];
        register float filter_w_2_1 = filter_buf[11];
        register float filter_w_2_2 = filter_buf[12];
        register float filter_w_2_3 = filter_buf[13];
        register float filter_w_2_4 = filter_buf[14];
        asm volatile("": : :"memory");

        // pass filter along
        if (!(tile_append & 1)) {
          // wait until remote filter buffer is ready
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (filter_f_E)), 0);
          //bsg_printf(" -- -- next filter buffer ready\n");
          spm_cpy<FILTER_BUF_SIZE>(filter_buf_remote, filter_buf);
          asm volatile("": : :"memory");
          *filter_f_E = 1;
          *filter_f_E_r = 1;
        }

        size_t total_images = N;
        if (filters == 0) {
          total_images += IMAGES_PER_BURST;
        }

        for (size_t images = 0; images < total_images; images += IMAGES_PER_BURST) {

          // wait until imap buf is filled
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (imap_f)), 1);

          // pass imap along
          if (!(tile_append & 2)) {
            // wait until remote imap buffer is ready
            bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (imap_f_NE)), 0);
            spm_cpy<IMAP_BUF_SIZE>(imap_buf_remote, imap_buf);
            asm volatile("": : :"memory");
            *imap_f_NE = 1;
            *imap_f_NE_r = 1;
          }

          // wait until psum buf is filled
          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f)), 1);
          //bsg_printf(" -- psum buffer filled\n");

          bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f_N)), 0);

          size_t   imap_offset = 0;
          size_t   psum_offset = 0;
          for (size_t image_id = 0; image_id < IMAGES_PER_BURST; image_id++) {
            /* This is the general functional code
            size_t filter_offset = 0;
            for (size_t filter_id = 0; filter_id < FILTERS_PER_PROCESSING_PASS; filter_id++) {
              // conv 1d -- just meant to be functional
              // sliding window
              for (size_t window = 0; window < Wout; window++) {
                // dot product between window and filter
                for (size_t filter_weight = 0; filter_weight < Wk; filter_weight++) {
                  psum_buf[psum_offset + window] +=
                    imap_buf[imap_offset + window + filter_weight] * filter_buf[filter_offset + filter_weight];
                }
              }
              psum_offset += Wout;
              filter_offset += Wk;
            }
            */
            // unroll the inner loop by 4, so we can simulate the ring buffer
            for (size_t window = 0; window < Wout; window += 4) {
              // load psum for (this window, 3 filters)
              register float imap_0 = imap_buf[imap_offset + window + 0];
              register float imap_1 = imap_buf[imap_offset + window + 1];
              register float imap_2 = imap_buf[imap_offset + window + 2];
              register float imap_3 = imap_buf[imap_offset + window + 3];
              register float imap_4 = imap_buf[imap_offset + window + 4];
              register float imap_5 = imap_buf[imap_offset + window + 5];
              register float imap_6 = imap_buf[imap_offset + window + 6];
              register float imap_7 = imap_buf[imap_offset + window + 7];

              register float psum_00 = psum_buf[psum_offset + window + 0];
              register float psum_10 = psum_buf[psum_offset + window + 28]; // hacky -- we know it's 28 at compile time
              register float psum_20 = psum_buf[psum_offset + window + 56];

              register float psum_01 = psum_buf[psum_offset + window + 1];
              register float psum_11 = psum_buf[psum_offset + window + 29];
              register float psum_21 = psum_buf[psum_offset + window + 57];
              asm volatile("": : :"memory");

              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_00) : "f"(imap_0), "f"(filter_w_0_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_10) : "f"(imap_0), "f"(filter_w_1_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_20) : "f"(imap_0), "f"(filter_w_2_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_00) : "f"(imap_1), "f"(filter_w_0_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_10) : "f"(imap_1), "f"(filter_w_1_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_20) : "f"(imap_1), "f"(filter_w_2_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_00) : "f"(imap_2), "f"(filter_w_0_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_10) : "f"(imap_2), "f"(filter_w_1_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_20) : "f"(imap_2), "f"(filter_w_2_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_00) : "f"(imap_3), "f"(filter_w_0_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_10) : "f"(imap_3), "f"(filter_w_1_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_20) : "f"(imap_3), "f"(filter_w_2_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_00) : "f"(imap_4), "f"(filter_w_0_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_10) : "f"(imap_4), "f"(filter_w_1_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_20) : "f"(imap_4), "f"(filter_w_2_4));

              // write back
              psum_buf_remote[psum_offset + window +  0] = psum_00;
              psum_buf_remote[psum_offset + window + 28] = psum_10;
              psum_buf_remote[psum_offset + window + 56] = psum_20;

              register float psum_02 = psum_buf[psum_offset + window + 2];
              register float psum_12 = psum_buf[psum_offset + window + 30];
              register float psum_22 = psum_buf[psum_offset + window + 58];
              asm volatile("": : :"memory");

              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_01) : "f"(imap_1), "f"(filter_w_0_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_11) : "f"(imap_1), "f"(filter_w_1_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_21) : "f"(imap_1), "f"(filter_w_2_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_01) : "f"(imap_2), "f"(filter_w_0_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_11) : "f"(imap_2), "f"(filter_w_1_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_21) : "f"(imap_2), "f"(filter_w_2_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_01) : "f"(imap_3), "f"(filter_w_0_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_11) : "f"(imap_3), "f"(filter_w_1_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_21) : "f"(imap_3), "f"(filter_w_2_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_01) : "f"(imap_4), "f"(filter_w_0_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_11) : "f"(imap_4), "f"(filter_w_1_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_21) : "f"(imap_4), "f"(filter_w_2_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_01) : "f"(imap_5), "f"(filter_w_0_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_11) : "f"(imap_5), "f"(filter_w_1_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_21) : "f"(imap_5), "f"(filter_w_2_4));

              // write back
              psum_buf_remote[psum_offset + window +  1] = psum_01;
              psum_buf_remote[psum_offset + window + 29] = psum_11;
              psum_buf_remote[psum_offset + window + 57] = psum_21;

              register float psum_03 = psum_buf[psum_offset + window + 3];
              register float psum_13 = psum_buf[psum_offset + window + 31];
              register float psum_23 = psum_buf[psum_offset + window + 59];
              asm volatile("": : :"memory");

              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_02) : "f"(imap_2), "f"(filter_w_0_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_12) : "f"(imap_2), "f"(filter_w_1_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_22) : "f"(imap_2), "f"(filter_w_2_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_02) : "f"(imap_3), "f"(filter_w_0_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_12) : "f"(imap_3), "f"(filter_w_1_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_22) : "f"(imap_3), "f"(filter_w_2_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_02) : "f"(imap_4), "f"(filter_w_0_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_12) : "f"(imap_4), "f"(filter_w_1_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_22) : "f"(imap_4), "f"(filter_w_2_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_02) : "f"(imap_5), "f"(filter_w_0_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_12) : "f"(imap_5), "f"(filter_w_1_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_22) : "f"(imap_5), "f"(filter_w_2_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_02) : "f"(imap_6), "f"(filter_w_0_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_12) : "f"(imap_6), "f"(filter_w_1_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_22) : "f"(imap_6), "f"(filter_w_2_4));

              // write back
              psum_buf_remote[psum_offset + window +  2] = psum_02;
              psum_buf_remote[psum_offset + window + 30] = psum_12;
              psum_buf_remote[psum_offset + window + 58] = psum_22;

              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_03) : "f"(imap_3), "f"(filter_w_0_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_13) : "f"(imap_3), "f"(filter_w_1_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_23) : "f"(imap_3), "f"(filter_w_2_0));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_03) : "f"(imap_4), "f"(filter_w_0_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_13) : "f"(imap_4), "f"(filter_w_1_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_23) : "f"(imap_4), "f"(filter_w_2_1));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_03) : "f"(imap_5), "f"(filter_w_0_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_13) : "f"(imap_5), "f"(filter_w_1_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_23) : "f"(imap_5), "f"(filter_w_2_2));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_03) : "f"(imap_6), "f"(filter_w_0_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_13) : "f"(imap_6), "f"(filter_w_1_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_23) : "f"(imap_6), "f"(filter_w_2_3));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_03) : "f"(imap_7), "f"(filter_w_0_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_13) : "f"(imap_7), "f"(filter_w_1_4));
              asm volatile("fmadd.s %0, %1, %2, %0" : "+f"(psum_23) : "f"(imap_7), "f"(filter_w_2_4));

              // write back
              psum_buf_remote[psum_offset + window +  3] = psum_03;
              psum_buf_remote[psum_offset + window + 31] = psum_13;
              psum_buf_remote[psum_offset + window + 59] = psum_23;
            }
            psum_offset += Wout * FILTERS_PER_PROCESSING_PASS;
            imap_offset += Win;
          }

          // signal imap free
          asm volatile("": : :"memory");
          *imap_f = 0;
          *imap_f_SW_r = 0;

          // pass psum along
          // wait until remote psum buffer is ready
          // bsg_wait_local(reinterpret_cast<int *> (const_cast<unsigned int*> (psum_f_N)), 0);
          // spm_cpy<PSUM_BUF_SIZE>(psum_buf_remote, psum_buf);
          asm volatile("": : :"memory");
          *psum_f_N = 1;
          *psum_f_N_r = 1;

          // signal psum and imap free
          asm volatile("": : :"memory");
          *psum_f = 0;
          *psum_f_S_r = 0;

          // switch buffer
          if (buffer_A) {
            psum_buf        = psum_buf_B;
            psum_f          = &psum_B_f;
            psum_f_S_r      = psum_B_f_S_r;
            psum_buf_remote = psum_buf_B_remote;
            psum_f_N        = &psum_B_f_N;
            psum_f_N_r      = psum_B_f_N_r;
            imap_buf        = imap_buf_B;
            imap_f          = &imap_B_f;
            imap_f_SW_r     = imap_B_f_SW_r;
            imap_buf_remote = imap_buf_B_remote;
            imap_f_NE       = &imap_B_f_NE;
            imap_f_NE_r     = imap_B_f_NE_r;
          } else {
            psum_buf        = psum_buf_A;
            psum_f          = &psum_A_f;
            psum_f_S_r      = psum_A_f_S_r;
            psum_buf_remote = psum_buf_A_remote;
            psum_f_N        = &psum_A_f_N;
            psum_f_N_r      = psum_A_f_N_r;
            imap_buf        = imap_buf_A;
            imap_f          = &imap_A_f;
            imap_f_SW_r     = imap_A_f_SW_r;
            imap_buf_remote = imap_buf_A_remote;
            imap_f_NE       = &imap_A_f_NE;
            imap_f_NE_r     = imap_A_f_NE_r;
          }
          buffer_A = !buffer_A;
        }
        // signal filter free
        asm volatile("": : :"memory");
        *filter_f = 0;
        *filter_f_W_r = 0;

        // switch buffer
        if (filter_buf == filter_buf_A) {
          filter_buf        = filter_buf_B;
          filter_f          = &filter_B_f;
          filter_f_W_r      = filter_B_f_W_r;
          filter_buf_remote = filter_buf_B_remote;
          filter_f_E        = &filter_B_f_E;
          filter_f_E_r      = filter_B_f_E_r;
        } else {
          filter_buf        = filter_buf_A;
          filter_f          = &filter_A_f;
          filter_f_W_r      = filter_A_f_W_r;
          filter_buf_remote = filter_buf_A_remote;
          filter_f_E        = &filter_A_f_E;
          filter_f_E_r      = filter_A_f_E_r;
        }
        // std::cout << " -- end of a pass -- " << std::endl;
      }
    };

    // main loop entrance
    g_barrier.sync();
    bsg_cuda_print_stat_kernel_start();

    // tile task dispatch
    switch (tile_config) {
      case 0:
        // nothing
        break;
      case 1:
        // filter DMA
        filterDMA();
        break;
      case 2:
        // imap DMA
        imapDMA();
        break;
      case 3:
        // psum DMA
        psumDMA();
        break;
      case 4:
        // compute
        computePE();
        break;
      case 5:
        // psum write back DMA
        psumWBDMA();
        break;
      default:
        hb_assert_msg(false, "invalid tile task config");
    }

    bsg_cuda_print_stat_kernel_end();

    g_barrier.sync();
    return 0;
  }

  HB_EMUL_REG_KERNEL(tensorlib_eyeriss, hb_tensor_t*, hb_tensor_t*, hb_tensor_t*,
                     hb_vector_t*, hb_vector_t*)

}
