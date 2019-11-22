//
//  vert_conv.cpp
//  18645_Project
//
//  Created by Steven Liu on 11/15/19.
//  Copyright © 2019 Steven Liu. All rights reserved.
//
#include <iostream>
#include <immintrin.h>
#include <math.h>
#include "vert_conv.hpp"

#define load_kernel_vertical_1(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \

#define load_kernel_vertical_2(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \

#define load_kernel_vertical_3(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \
k2 = _mm256_broadcast_ss(k_ptr + 2); \

#define load_kernel_vertical_4(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \
k2 = _mm256_broadcast_ss(k_ptr + 2); \
k3 = _mm256_broadcast_ss(k_ptr + 3); \

#define load_kernel_vertical_5(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \
k2 = _mm256_broadcast_ss(k_ptr + 2); \
k3 = _mm256_broadcast_ss(k_ptr + 3); \
k4 = _mm256_broadcast_ss(k_ptr + 4); \

#define load_kernel_vertical_6(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \
k2 = _mm256_broadcast_ss(k_ptr + 2); \
k3 = _mm256_broadcast_ss(k_ptr + 3); \
k4 = _mm256_broadcast_ss(k_ptr + 4); \
k5 = _mm256_broadcast_ss(k_ptr + 5); \

#define load_kernel_vertical_7(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \
k2 = _mm256_broadcast_ss(k_ptr + 2); \
k3 = _mm256_broadcast_ss(k_ptr + 3); \
k4 = _mm256_broadcast_ss(k_ptr + 4); \
k5 = _mm256_broadcast_ss(k_ptr + 5); \
k6 = _mm256_broadcast_ss(k_ptr + 6); \
k7 = _mm256_broadcast_ss(k_ptr + 7); \

#define load_kernel_vertical(k_ptr, k0, k1, k2, k3, k4, k5, k6, k7) \
k0 = _mm256_broadcast_ss(k_ptr + 0); \
k1 = _mm256_broadcast_ss(k_ptr + 1); \
k2 = _mm256_broadcast_ss(k_ptr + 2); \
k3 = _mm256_broadcast_ss(k_ptr + 3); \
k4 = _mm256_broadcast_ss(k_ptr + 4); \
k5 = _mm256_broadcast_ss(k_ptr + 5); \
k6 = _mm256_broadcast_ss(k_ptr + 6); \
k7 = _mm256_broadcast_ss(k_ptr + 7); \

#define load_img_vertical_1(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \

#define load_img_vertical_2(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \

#define load_img_vertical_3(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \

#define load_img_vertical_4(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \
s3 = _mm256_loadu_ps(src_ptr + 3 * n_col); \

#define load_img_vertical_5(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \
s3 = _mm256_loadu_ps(src_ptr + 3 * n_col); \
s4 = _mm256_loadu_ps(src_ptr + 4 * n_col); \

#define load_img_vertical_6(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \
s3 = _mm256_loadu_ps(src_ptr + 3 * n_col); \
s4 = _mm256_loadu_ps(src_ptr + 4 * n_col); \
s5 = _mm256_loadu_ps(src_ptr + 5 * n_col); \

#define load_img_vertical_7(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \
s3 = _mm256_loadu_ps(src_ptr + 3 * n_col); \
s4 = _mm256_loadu_ps(src_ptr + 4 * n_col); \
s5 = _mm256_loadu_ps(src_ptr + 5 * n_col); \
s6 = _mm256_loadu_ps(src_ptr + 6 * n_col); \

#define load_img_vertical(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7) \
s0 = _mm256_loadu_ps(src_ptr + 0 * n_col); \
s1 = _mm256_loadu_ps(src_ptr + 1 * n_col); \
s2 = _mm256_loadu_ps(src_ptr + 2 * n_col); \
s3 = _mm256_loadu_ps(src_ptr + 3 * n_col); \
s4 = _mm256_loadu_ps(src_ptr + 4 * n_col); \
s5 = _mm256_loadu_ps(src_ptr + 5 * n_col); \
s6 = _mm256_loadu_ps(src_ptr + 6 * n_col); \
s7 = _mm256_loadu_ps(src_ptr + 7 * n_col); \

#define kernel_veritcal_1(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_1(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \

#define kernel_veritcal_2(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_2(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s0 = _mm256_add_ps(s0, s1); \

#define kernel_veritcal_3(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_3(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s2 = _mm256_mul_ps(s2, k2); \
s0 = _mm256_add_ps(s0, s1); \
s0 = _mm256_add_ps(s0, s2); \

#define kernel_veritcal_4(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_4(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s2 = _mm256_mul_ps(s2, k2); \
s3 = _mm256_mul_ps(s3, k3); \
s0 = _mm256_add_ps(s0, s1); \
s2 = _mm256_add_ps(s2, s3); \
s0 = _mm256_add_ps(s0, s2); \

#define kernel_veritcal_5(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_5(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s2 = _mm256_mul_ps(s2, k2); \
s3 = _mm256_mul_ps(s3, k3); \
s4 = _mm256_mul_ps(s4, k4); \
s0 = _mm256_add_ps(s0, s1); \
s2 = _mm256_add_ps(s2, s3); \
s0 = _mm256_add_ps(s0, s2); \
s0 = _mm256_add_ps(s0, s4); \

#define kernel_veritcal_6(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_6(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s2 = _mm256_mul_ps(s2, k2); \
s3 = _mm256_mul_ps(s3, k3); \
s4 = _mm256_mul_ps(s4, k4); \
s5 = _mm256_mul_ps(s5, k5); \
s0 = _mm256_add_ps(s0, s1); \
s2 = _mm256_add_ps(s2, s3); \
s4 = _mm256_add_ps(s4, s5); \
s0 = _mm256_add_ps(s0, s2); \
s0 = _mm256_add_ps(s0, s4); \

#define kernel_veritcal_7(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical_7(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s2 = _mm256_mul_ps(s2, k2); \
s3 = _mm256_mul_ps(s3, k3); \
s4 = _mm256_mul_ps(s4, k4); \
s5 = _mm256_mul_ps(s5, k5); \
s6 = _mm256_mul_ps(s6, k6); \
s0 = _mm256_add_ps(s0, s1); \
s2 = _mm256_add_ps(s2, s3); \
s4 = _mm256_add_ps(s4, s5); \
s0 = _mm256_add_ps(s0, s2); \
s4 = _mm256_add_ps(s4, s6); \
s0 = _mm256_add_ps(s0, s4); \

#define kernel_veritcal(src_ptr, n_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7) \
load_img_vertical(src_ptr, n_col, s0, s1, s2, s3, s4, s5, s6, s7); \
s0 = _mm256_mul_ps(s0, k0); \
s1 = _mm256_mul_ps(s1, k1); \
s2 = _mm256_mul_ps(s2, k2); \
s3 = _mm256_mul_ps(s3, k3); \
s4 = _mm256_mul_ps(s4, k4); \
s5 = _mm256_mul_ps(s5, k5); \
s6 = _mm256_mul_ps(s6, k6); \
s7 = _mm256_mul_ps(s7, k7); \
s0 = _mm256_add_ps(s0, s1); \
s2 = _mm256_add_ps(s2, s3); \
s4 = _mm256_add_ps(s4, s5); \
s6 = _mm256_add_ps(s6, s7); \
s0 = _mm256_add_ps(s0, s2); \
s4 = _mm256_add_ps(s4, s6); \
s0 = _mm256_add_ps(s0, s4); \

#define sum_to_d(d0, s0) \
d0 = _mm256_add_ps(d0, s0); \

#define store_d_vertical(dst_ptr, d0, d1) \
*(dst_ptr + 0) = d0[0]; \
*(dst_ptr + 1) = d0[1]; \
*(dst_ptr + 2) = d0[2]; \
*(dst_ptr + 3) = d0[3]; \
*(dst_ptr + 4) = d0[4]; \
*(dst_ptr + 5) = d0[5]; \
*(dst_ptr + 6) = d0[6]; \
*(dst_ptr + 7) = d0[7]; \
*(dst_ptr + 8) = d1[0]; \
*(dst_ptr + 9) = d1[1]; \
*(dst_ptr + 10) = d1[2]; \
*(dst_ptr + 11) = d1[3]; \
*(dst_ptr + 12) = d1[4]; \
*(dst_ptr + 13) = d1[5]; \
*(dst_ptr + 14) = d1[6]; \
*(dst_ptr + 15) = d1[7]; \

void vertical_kernel_conv(int src_row, int src_col, float* src_ptr, int dst_row, int dst_col, float* dst_ptr, int ksize, const float* k_ptr) {
    const int num_of_simd_for_one_kernel = (int)ceil(((double)ksize)/8.0);
//    printf("num_of_simd_for_one_kernel = %i\n", num_of_simd_for_one_kernel);
    
    //    const int half_ksize = ksize/2;
    const int partial_SIMD_num = ksize - 8 * (num_of_simd_for_one_kernel - 1);
//    printf("partial_SIMD_num = %i\n", partial_SIMD_num);
    
    __m256 k0, k1, k2, k3, k4, k5, k6, k7;
    __m256 s0, s1, s2, s3, s4, s5, s6, s7;
    __m256 d0, d1;
    
    int i, j, k;
    for (i = 0; i < dst_row; i += 1) {
        // printf("i:   %i\n\n", i);
        for (j = 0; j < dst_col; j += 16) {
            // printf("j:   %i\n\n", j);
            
            // computational kernel
            d0 = _mm256_setzero_ps();
            d1 = _mm256_setzero_ps();
            // full SIMDs
            for (k = 0; k < num_of_simd_for_one_kernel - 1; k++) {
                load_kernel_vertical(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                kernel_veritcal(src_ptr+i*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                sum_to_d(d0, s0);
                kernel_veritcal(src_ptr+i*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                sum_to_d(d1, s0);
            }
            
            
            // partical SIMDs
//            printf("k: %i\n",k);
            switch (partial_SIMD_num) {
                case 1:
                    load_kernel_vertical_1(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_1(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_1(src_ptr+(i+k*8)*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 2:
                    load_kernel_vertical_2(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_2(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_2(src_ptr+(i+k*8)*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 3:
                    load_kernel_vertical_3(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_3(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_3(src_ptr+(i+k*8)*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 4:
                    load_kernel_vertical_4(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_4(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_4(src_ptr+(i+k*8)*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 5:
                    load_kernel_vertical_5(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_5(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_5(src_ptr+(i+k*8)*dst_col+j*dst_col+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                    
                case 6:
                    load_kernel_vertical_6(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_6(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_6(src_ptr+(i+k*8)*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                    
                case 7:
                    load_kernel_vertical_7(k_ptr + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_7(src_ptr+(i+k*8)*dst_col+j, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_7(src_ptr+(i+k*8)*dst_col+j+8, src_col, k_ptr, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                    
                default:
                    break;
            }
            store_d_vertical(dst_ptr+i*dst_col+j, d0, d1);
        }
    }
}
