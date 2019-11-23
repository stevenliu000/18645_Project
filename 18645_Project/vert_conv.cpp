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
#include <string>
#include <fstream>

#define load_kernel_vertical_1(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \

#define load_kernel_vertical_2(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \

#define load_kernel_vertical_3(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \
k2_v = _mm256_broadcast_ss(k_ptr_v + 2); \

#define load_kernel_vertical_4(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \
k2_v = _mm256_broadcast_ss(k_ptr_v + 2); \
k3_v = _mm256_broadcast_ss(k_ptr_v + 3); \

#define load_kernel_vertical_5(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \
k2_v = _mm256_broadcast_ss(k_ptr_v + 2); \
k3_v = _mm256_broadcast_ss(k_ptr_v + 3); \
k4_v = _mm256_broadcast_ss(k_ptr_v + 4); \

#define load_kernel_vertical_6(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \
k2_v = _mm256_broadcast_ss(k_ptr_v + 2); \
k3_v = _mm256_broadcast_ss(k_ptr_v + 3); \
k4_v = _mm256_broadcast_ss(k_ptr_v + 4); \
k5_v = _mm256_broadcast_ss(k_ptr_v + 5); \

#define load_kernel_vertical_7(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \
k2_v = _mm256_broadcast_ss(k_ptr_v + 2); \
k3_v = _mm256_broadcast_ss(k_ptr_v + 3); \
k4_v = _mm256_broadcast_ss(k_ptr_v + 4); \
k5_v = _mm256_broadcast_ss(k_ptr_v + 5); \
k6_v = _mm256_broadcast_ss(k_ptr_v + 6); \
k7_v = _mm256_broadcast_ss(k_ptr_v + 7); \

#define load_kernel_vertical(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_broadcast_ss(k_ptr_v + 0); \
k1_v = _mm256_broadcast_ss(k_ptr_v + 1); \
k2_v = _mm256_broadcast_ss(k_ptr_v + 2); \
k3_v = _mm256_broadcast_ss(k_ptr_v + 3); \
k4_v = _mm256_broadcast_ss(k_ptr_v + 4); \
k5_v = _mm256_broadcast_ss(k_ptr_v + 5); \
k6_v = _mm256_broadcast_ss(k_ptr_v + 6); \
k7_v = _mm256_broadcast_ss(k_ptr_v + 7); \

#define load_img_vertical_1(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \

#define load_img_vertical_2(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \

#define load_img_vertical_3(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \
s2_v = _mm256_load_ps(src_ptr_v + 2 * n_col_v); \

#define load_img_vertical_4(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \
s2_v = _mm256_load_ps(src_ptr_v + 2 * n_col_v); \
s3_v = _mm256_load_ps(src_ptr_v + 3 * n_col_v); \

#define load_img_vertical_5(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \
s2_v = _mm256_load_ps(src_ptr_v + 2 * n_col_v); \
s3_v = _mm256_load_ps(src_ptr_v + 3 * n_col_v); \
s4_v = _mm256_load_ps(src_ptr_v + 4 * n_col_v); \

#define load_img_vertical_6(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \
s2_v = _mm256_load_ps(src_ptr_v + 2 * n_col_v); \
s3_v = _mm256_load_ps(src_ptr_v + 3 * n_col_v); \
s4_v = _mm256_load_ps(src_ptr_v + 4 * n_col_v); \
s5_v = _mm256_load_ps(src_ptr_v + 5 * n_col_v); \

#define load_img_vertical_7(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \
s2_v = _mm256_load_ps(src_ptr_v + 2 * n_col_v); \
s3_v = _mm256_load_ps(src_ptr_v + 3 * n_col_v); \
s4_v = _mm256_load_ps(src_ptr_v + 4 * n_col_v); \
s5_v = _mm256_load_ps(src_ptr_v + 5 * n_col_v); \
s6_v = _mm256_load_ps(src_ptr_v + 6 * n_col_v); \

#define load_img_vertical(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + 0 * n_col_v); \
s1_v = _mm256_load_ps(src_ptr_v + 1 * n_col_v); \
s2_v = _mm256_load_ps(src_ptr_v + 2 * n_col_v); \
s3_v = _mm256_load_ps(src_ptr_v + 3 * n_col_v); \
s4_v = _mm256_load_ps(src_ptr_v + 4 * n_col_v); \
s5_v = _mm256_load_ps(src_ptr_v + 5 * n_col_v); \
s6_v = _mm256_load_ps(src_ptr_v + 6 * n_col_v); \
s7_v = _mm256_load_ps(src_ptr_v + 7 * n_col_v); \

#define kernel_veritcal_1(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_1(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \

#define kernel_veritcal_2(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_2(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \

#define kernel_veritcal_3(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_3(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s2_v = _mm256_mul_ps(s2_v, k2_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \
s0_v = _mm256_add_ps(s0_v, s2_v); \

#define kernel_veritcal_4(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_4(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s2_v = _mm256_mul_ps(s2_v, k2_v); \
s3_v = _mm256_mul_ps(s3_v, k3_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \
s2_v = _mm256_add_ps(s2_v, s3_v); \
s0_v = _mm256_add_ps(s0_v, s2_v); \

#define kernel_veritcal_5(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_5(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s2_v = _mm256_mul_ps(s2_v, k2_v); \
s3_v = _mm256_mul_ps(s3_v, k3_v); \
s4_v = _mm256_mul_ps(s4_v, k4_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \
s2_v = _mm256_add_ps(s2_v, s3_v); \
s0_v = _mm256_add_ps(s0_v, s2_v); \
s0_v = _mm256_add_ps(s0_v, s4_v); \

#define kernel_veritcal_6(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_6(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s2_v = _mm256_mul_ps(s2_v, k2_v); \
s3_v = _mm256_mul_ps(s3_v, k3_v); \
s4_v = _mm256_mul_ps(s4_v, k4_v); \
s5_v = _mm256_mul_ps(s5_v, k5_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \
s2_v = _mm256_add_ps(s2_v, s3_v); \
s4_v = _mm256_add_ps(s4_v, s5_v); \
s0_v = _mm256_add_ps(s0_v, s2_v); \
s0_v = _mm256_add_ps(s0_v, s4_v); \

#define kernel_veritcal_7(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical_7(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s2_v = _mm256_mul_ps(s2_v, k2_v); \
s3_v = _mm256_mul_ps(s3_v, k3_v); \
s4_v = _mm256_mul_ps(s4_v, k4_v); \
s5_v = _mm256_mul_ps(s5_v, k5_v); \
s6_v = _mm256_mul_ps(s6_v, k6_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \
s2_v = _mm256_add_ps(s2_v, s3_v); \
s4_v = _mm256_add_ps(s4_v, s5_v); \
s0_v = _mm256_add_ps(s0_v, s2_v); \
s4_v = _mm256_add_ps(s4_v, s6_v); \
s0_v = _mm256_add_ps(s0_v, s4_v); \

#define kernel_veritcal(src_ptr_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
load_img_vertical(src_ptr_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
s0_v = _mm256_mul_ps(s0_v, k0_v); \
s1_v = _mm256_mul_ps(s1_v, k1_v); \
s2_v = _mm256_mul_ps(s2_v, k2_v); \
s3_v = _mm256_mul_ps(s3_v, k3_v); \
s4_v = _mm256_mul_ps(s4_v, k4_v); \
s5_v = _mm256_mul_ps(s5_v, k5_v); \
s6_v = _mm256_mul_ps(s6_v, k6_v); \
s7_v = _mm256_mul_ps(s7_v, k7_v); \
s0_v = _mm256_add_ps(s0_v, s1_v); \
s2_v = _mm256_add_ps(s2_v, s3_v); \
s4_v = _mm256_add_ps(s4_v, s5_v); \
s6_v = _mm256_add_ps(s6_v, s7_v); \
s0_v = _mm256_add_ps(s0_v, s2_v); \
s4_v = _mm256_add_ps(s4_v, s6_v); \
s0_v = _mm256_add_ps(s0_v, s4_v); \

#define sum_to_d(d0_v, s0_v) \
d0_v = _mm256_add_ps(d0_v, s0_v); \

#define store_d_vertical(dst_ptr_v, d0_v, d1_v) \
*(dst_ptr_v + 0) = d0_v[0]; \
*(dst_ptr_v + 1) = d0_v[1]; \
*(dst_ptr_v + 2) = d0_v[2]; \
*(dst_ptr_v + 3) = d0_v[3]; \
*(dst_ptr_v + 4) = d0_v[4]; \
*(dst_ptr_v + 5) = d0_v[5]; \
*(dst_ptr_v + 6) = d0_v[6]; \
*(dst_ptr_v + 7) = d0_v[7]; \
*(dst_ptr_v + 8) = d1_v[0]; \
*(dst_ptr_v + 9) = d1_v[1]; \
*(dst_ptr_v + 10) = d1_v[2]; \
*(dst_ptr_v + 11) = d1_v[3]; \
*(dst_ptr_v + 12) = d1_v[4]; \
*(dst_ptr_v + 13) = d1_v[5]; \
*(dst_ptr_v + 14) = d1_v[6]; \
*(dst_ptr_v + 15) = d1_v[7]; \

void vertical_kernel_conv(int src_row, int src_col, float* src_ptr_v, int dst_row, int dst_col, float* dst_ptr_v, int ksize, const float* k_ptr_v) {
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
                load_kernel_vertical(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                kernel_veritcal(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                sum_to_d(d0, s0);
                kernel_veritcal(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                sum_to_d(d1, s0);
            }
            
            
            // partical SIMDs
            //            printf("k: %i\n",k);
            switch (partial_SIMD_num) {
                case 1:
                    load_kernel_vertical_1(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_1(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_1(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 2:
                    load_kernel_vertical_2(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_2(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_2(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 3:
                    load_kernel_vertical_3(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_3(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_3(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 4:
                    load_kernel_vertical_4(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_4(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_4(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                case 5:
                    load_kernel_vertical_5(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_5(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_5(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                    
                case 6:
                    load_kernel_vertical_6(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_6(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_6(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                    
                case 7:
                    load_kernel_vertical_7(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_veritcal_7(src_ptr_v+(i+k*8)*dst_col+j, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                    kernel_veritcal_7(src_ptr_v+(i+k*8)*dst_col+j+8, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d1, s0);
                    break;
                    
                default:
                    break;
            }
            store_d_vertical(dst_ptr_v+i*dst_col+j, d0, d1);
        }
    }
}
