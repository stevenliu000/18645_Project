//
//  hor_conv.cpp
//  18645_Project
//
//  Created by Steven Liu on 11/15/19.
//  Copyright © 2019 Steven Liu. All rights reserved.
//

#include "hor_conv.hpp"
#include <iostream>
#include <immintrin.h>
#include <math.h>

#define load_img_horizontal(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \
s3 = _mm256_loadu_ps(src_ptr + 3 * n_col); \
s4 = _mm256_loadu_ps(src_ptr + 4 * n_col); \
s5 = _mm256_loadu_ps(src_ptr + 5 * n_col); \
s6 = _mm256_loadu_ps(src_ptr + 6 * n_col); \
s7 = _mm256_loadu_ps(src_ptr + 7 * n_col); \

#define load_kernel_horizontal(l_ptr, k) \
k = _mm256_loadu_ps(l_ptr); \

#define mul_horizontal(s0, s1, s2, s3, s4, s5, s6, s7, k) \
s0 = _mm256_mul_ps(s0, k); \
s1 = _mm256_mul_ps(s1, k); \
s2 = _mm256_mul_ps(s2, k); \
s3 = _mm256_mul_ps(s3, k); \
s4 = _mm256_mul_ps(s4, k); \
s5 = _mm256_mul_ps(s5, k); \
s6 = _mm256_mul_ps(s6, k); \
s7 = _mm256_mul_ps(s7, k); \

#define reduce_horizontal(s0, s1, s2, s3, s4, s5, s6, s7, h1, h2, h3, h4, h5, h6, d0) \
h1 = _mm256_hadd_ps(s0, s1); \
h4 = _mm256_hadd_ps(s4, s5); \
h2 = _mm256_hadd_ps(s2, s3); \
h5 = _mm256_hadd_ps(s6, s7); \
h3 = _mm256_hadd_ps(h1, h2); \
h6 = _mm256_hadd_ps(h4, h5); \
d0[0] += h3[0] + h3[4]; \
d0[1] += h3[1] + h3[5]; \
d0[2] += h3[2] + h3[6]; \
d0[3] += h3[3] + h3[7]; \
d0[4] += h6[0] + h6[4]; \
d0[5] += h6[1] + h6[5]; \
d0[6] += h6[2] + h6[6]; \
d0[7] += h6[3] + h6[7]; \

#define store_d_horizantal(d0, dst_prt_tmp, n_col) \
*(dst_prt_tmp + 0 * n_col) = d0[0]; \
*(dst_prt_tmp + 1 * n_col) = d0[1]; \
*(dst_prt_tmp + 2 * n_col) = d0[2]; \
*(dst_prt_tmp + 3 * n_col) = d0[3]; \
*(dst_prt_tmp + 4 * n_col) = d0[4]; \
*(dst_prt_tmp + 5 * n_col) = d0[5]; \
*(dst_prt_tmp + 6 * n_col) = d0[6]; \
*(dst_prt_tmp + 7 * n_col) = d0[7]; \


#define horizontal_kernel(src_ptr, n_col, l_ptr, k, s0, s1, s2, s3, s4, s5, s6, s7, h1, h2, h3, h4, h5, h6, dst) \
load_img_horizontal(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
load_kernel_horizontal(l_ptr, k) \
mul_horizontal(s0, s1, s2, s3, s4, s5, s6, s7, k) \
reduce_horizontal(s0, s1, s2, s3, s4, s5, s6, s7, h1, h2, h3, h4, h5, h6, dst) \

void horizontal_kernel_conv(int src_row, int src_col, const float* src_ptr, int dst_row, int dst_col, float* dst_ptr, int ksize, const float* k_ptr) {
    int i, j, k;
    const int num_of_simd_for_one_kernel = (int)ceil(((double)ksize)/8.0);
//    printf("num_of_simd_for_one_kernel = %i\n", num_of_simd_for_one_kernel);
    
    //    const int half_ksize = ksize/2;
    const int partial_SIMD_num = ksize - 8 * (num_of_simd_for_one_kernel - 1);
    
    float padded_kernel[num_of_simd_for_one_kernel*8];
    
    for (i = 0; i < num_of_simd_for_one_kernel*8; i++) {
        padded_kernel[i] = 0;
    }
    
    for (i = 0; i < ksize; i++) {
        padded_kernel[i] = *(k_ptr+i);
    }
    
    
//    printf("partial_SIMD_num = %i\n", partial_SIMD_num);
    
    __m256 h1, h2, h3, h4, h5, h6;
    __m256 s0, s1, s2, s3, s4, s5, s6, s7;
    __m256 d0;
    __m256 kernel_SIMD;
    
    for (i = 0; i < dst_row; i += 8) {
        // printf("i: %i\n",i);
        for (j = 0; j < dst_col; j += 1) {
            // printf("j: %i\n",j);
            
            // computational kernel
            d0 = _mm256_setzero_ps();
            // full SIMDs
            for (k = 0; k < num_of_simd_for_one_kernel; k++) {
                horizontal_kernel(src_ptr+i*src_col+j+k*8, src_col, (const float*)((&padded_kernel)+k*8), kernel_SIMD, s0, s1, s2, s3, s4, s5, s6, s7, h1, h2, h3, h4, h5, h6, d0);
            }
//            for (int cc = 0; cc<8; cc++) {
//                printf("%f ",d0[cc]);
//            }
//            printf("\n");
//            printf("here!!!!, %i\n\n\n", i*dst_col+j);
            store_d_horizantal(d0, dst_ptr+i*dst_col+j, dst_col);
        }
    }
}
