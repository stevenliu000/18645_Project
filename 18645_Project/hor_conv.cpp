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

using namespace std;

#define load_img_horizontal(src_ptr_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_loadu_ps(src_ptr_h + 0 * n_col_h); \
s1_h = _mm256_loadu_ps(src_ptr_h + 1 * n_col_h); \
s2_h = _mm256_loadu_ps(src_ptr_h + 2 * n_col_h); \
s3_h = _mm256_loadu_ps(src_ptr_h + 3 * n_col_h); \
s4_h = _mm256_loadu_ps(src_ptr_h + 4 * n_col_h); \
s5_h = _mm256_loadu_ps(src_ptr_h + 5 * n_col_h); \
s6_h = _mm256_loadu_ps(src_ptr_h + 6 * n_col_h); \
s7_h = _mm256_loadu_ps(src_ptr_h + 7 * n_col_h); \

#define load_kernel_horizontal(l_ptr_h, k_h) \
k_h = _mm256_loadu_ps(l_ptr_h); \

#define mul_horizontal(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k_h) \
s0_h = _mm256_mul_ps(s0_h, k_h); \
s1_h = _mm256_mul_ps(s1_h, k_h); \
s2_h = _mm256_mul_ps(s2_h, k_h); \
s3_h = _mm256_mul_ps(s3_h, k_h); \
s4_h = _mm256_mul_ps(s4_h, k_h); \
s5_h = _mm256_mul_ps(s5_h, k_h); \
s6_h = _mm256_mul_ps(s6_h, k_h); \
s7_h = _mm256_mul_ps(s7_h, k_h); \

#define reduce_horizontal(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, h1_h, h2_h, h3_h, h4_h, h5_h, h6_h, d0_h) \
h1_h = _mm256_hadd_ps(s0_h, s1_h); \
h4_h = _mm256_hadd_ps(s4_h, s5_h); \
h2_h = _mm256_hadd_ps(s2_h, s3_h); \
h5_h = _mm256_hadd_ps(s6_h, s7_h); \
h3_h = _mm256_hadd_ps(h1_h, h2_h); \
h6_h = _mm256_hadd_ps(h4_h, h5_h); \
d0_h[0] += h3_h[0] + h3_h[4]; \
d0_h[1] += h3_h[1] + h3_h[5]; \
d0_h[2] += h3_h[2] + h3_h[6]; \
d0_h[3] += h3_h[3] + h3_h[7]; \
d0_h[4] += h6_h[0] + h6_h[4]; \
d0_h[5] += h6_h[1] + h6_h[5]; \
d0_h[6] += h6_h[2] + h6_h[6]; \
d0_h[7] += h6_h[3] + h6_h[7]; \

#define store_d_horizantal(d0_h, dst_prt_tmp_h, n_col_h) \
*(dst_prt_tmp_h + 0 * n_col_h) = d0_h[0]; \
*(dst_prt_tmp_h + 1 * n_col_h) = d0_h[1]; \
*(dst_prt_tmp_h + 2 * n_col_h) = d0_h[2]; \
*(dst_prt_tmp_h + 3 * n_col_h) = d0_h[3]; \
*(dst_prt_tmp_h + 4 * n_col_h) = d0_h[4]; \
*(dst_prt_tmp_h + 5 * n_col_h) = d0_h[5]; \
*(dst_prt_tmp_h + 6 * n_col_h) = d0_h[6]; \
*(dst_prt_tmp_h + 7 * n_col_h) = d0_h[7]; \


#define horizontal_kernel(src_ptr_h, n_col_h, l_ptr_h, k_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, h1_h, h2_h, h3_h, h4_h, h5_h, h6_h, dst_h) \
load_img_horizontal(src_ptr_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
load_kernel_horizontal(l_ptr_h, k_h) \
mul_horizontal(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k_h) \
reduce_horizontal(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, h1_h, h2_h, h3_h, h4_h, h5_h, h6_h, dst_h) \

void horizontal_kernel_conv(int src_row, int src_col, const float* src_ptr_h, int dst_row, int dst_col, float* dst_ptr, int ksize, const float* k_ptr) {
    int i, j, k;
    const int num_of_simd_for_one_kernel = (int)ceil(((double)ksize)/8.0);
//    printf("num_of_simd_for_one_kernel = %i\n", num_of_simd_for_one_kernel);
    
    //    const int half_ksize = ksize/2;
//    const int partial_SIMD_num = ksize - 8 * (num_of_simd_for_one_kernel - 1);
    
    float padded_kernel[num_of_simd_for_one_kernel*8];
    
    for (i = 0; i < num_of_simd_for_one_kernel*8; i++) {
        padded_kernel[i] = 0;
    }
    
    for (i = 0; i < ksize; i++) {
        padded_kernel[i] = *(k_ptr+i);
    }
    
    
    __m256 h1_h, h2_h, h3_h, h4_h, h5_h, h6_h;
    __m256 s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h;
    __m256 d0_h;
    __m256 k_h;
    
    for (i = 0; i < dst_row; i += 8) {
        for (j = 0; j < dst_col; j += 1) {
            
            // computational kernel
            d0_h = _mm256_setzero_ps();
            // full SIMDs
            for (k = 0; k < num_of_simd_for_one_kernel; k++) {
                horizontal_kernel(src_ptr_h+i*src_col+j+k*8, src_col, (float*)(&padded_kernel[k*8]), k_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, h1_h, h2_h, h3_h, h4_h, h5_h, h6_h, d0_h);
                
//                printf("s0: %f %f %f %f %f %f %f %f\n", s0_h[0], s0_h[1], s0_h[2], s0_h[3], s0_h[4], s0_h[5], s0_h[6], s0_h[7]);
//                printf("s1: %f %f %f %f %f %f %f %f\n", s1_h[0], s1_h[1], s1_h[2], s1_h[3], s1_h[4], s1_h[5], s1_h[6], s1_h[7]);
            }
            store_d_horizantal(d0_h, dst_ptr+i*dst_col+j, dst_col);
        }
    }
}
