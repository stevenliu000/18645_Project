//
//  vert_conv_liu.cpp
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
#include "transpose.hpp"

#define load_kernel_horizontal_liu_1(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \

#define load_kernel_horizontal_liu_2(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \

#define load_kernel_horizontal_liu_3(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \
k2_h = _mm256_broadcast_ss(k_ptr_h + 2); \

#define load_kernel_horizontal_liu_4(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \
k2_h = _mm256_broadcast_ss(k_ptr_h + 2); \
k3_h = _mm256_broadcast_ss(k_ptr_h + 3); \

#define load_kernel_horizontal_liu_5(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \
k2_h = _mm256_broadcast_ss(k_ptr_h + 2); \
k3_h = _mm256_broadcast_ss(k_ptr_h + 3); \
k4_h = _mm256_broadcast_ss(k_ptr_h + 4); \

#define load_kernel_horizontal_liu_6(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \
k2_h = _mm256_broadcast_ss(k_ptr_h + 2); \
k3_h = _mm256_broadcast_ss(k_ptr_h + 3); \
k4_h = _mm256_broadcast_ss(k_ptr_h + 4); \
k5_h = _mm256_broadcast_ss(k_ptr_h + 5); \

#define load_kernel_horizontal_liu_7(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \
k2_h = _mm256_broadcast_ss(k_ptr_h + 2); \
k3_h = _mm256_broadcast_ss(k_ptr_h + 3); \
k4_h = _mm256_broadcast_ss(k_ptr_h + 4); \
k5_h = _mm256_broadcast_ss(k_ptr_h + 5); \
k6_h = _mm256_broadcast_ss(k_ptr_h + 6); \
k7_h = _mm256_broadcast_ss(k_ptr_h + 7); \

#define load_kernel_horizontal_liu(k_ptr_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
k0_h = _mm256_broadcast_ss(k_ptr_h + 0); \
k1_h = _mm256_broadcast_ss(k_ptr_h + 1); \
k2_h = _mm256_broadcast_ss(k_ptr_h + 2); \
k3_h = _mm256_broadcast_ss(k_ptr_h + 3); \
k4_h = _mm256_broadcast_ss(k_ptr_h + 4); \
k5_h = _mm256_broadcast_ss(k_ptr_h + 5); \
k6_h = _mm256_broadcast_ss(k_ptr_h + 6); \
k7_h = _mm256_broadcast_ss(k_ptr_h + 7); \

#define load_img_horizontal_liu_1(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu_2(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 1, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu_3(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 1, n_col_h, pad_size_h) * n_col_h + j); \
s2_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 2, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu_4(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 1, n_col_h, pad_size_h) * n_col_h + j); \
s2_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 2, n_col_h, pad_size_h) * n_col_h + j); \
s3_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 3, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu_5(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 1, n_col_h, pad_size_h) * n_col_h + j); \
s2_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 2, n_col_h, pad_size_h) * n_col_h + j); \
s3_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 3, n_col_h, pad_size_h) * n_col_h + j); \
s4_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 4, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu_6(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 1, n_col_h, pad_size_h) * n_col_h + j); \
s2_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 2, n_col_h, pad_size_h) * n_col_h + j); \
s3_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 3, n_col_h, pad_size_h) * n_col_h + j); \
s4_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 4, n_col_h, pad_size_h) * n_col_h + j); \
s5_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 5, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu_7(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 0, n_col_h, pad_size_h) * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 1, n_col_h, pad_size_h) * n_col_h + j); \
s2_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 2, n_col_h, pad_size_h) * n_col_h + j); \
s3_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 3, n_col_h, pad_size_h) * n_col_h + j); \
s4_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 4, n_col_h, pad_size_h) * n_col_h + j); \
s5_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 5, n_col_h, pad_size_h) * n_col_h + j); \
s6_h = _mm256_load_ps(src_ptr_h + index_transform(i_rows + 6, n_col_h, pad_size_h) * n_col_h + j); \

#define load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h) \
s0_h = _mm256_load_ps(src_ptr_h + i_rows + 0 * n_col_h + j); \
s1_h = _mm256_load_ps(src_ptr_h + i_rows + 1 * n_col_h + j); \
s2_h = _mm256_load_ps(src_ptr_h + i_rows + 2 * n_col_h + j); \
s3_h = _mm256_load_ps(src_ptr_h + i_rows + 3 * n_col_h + j); \
s4_h = _mm256_load_ps(src_ptr_h + i_rows + 4 * n_col_h + j); \
s5_h = _mm256_load_ps(src_ptr_h + i_rows + 5 * n_col_h + j); \
s6_h = _mm256_load_ps(src_ptr_h + i_rows + 6 * n_col_h + j); \
s7_h = _mm256_load_ps(src_ptr_h + i_rows + 7 * n_col_h + j); \

#define kernel_horizontal_liu_1(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \

#define kernel_horizontal_liu_2(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \

#define kernel_horizontal_liu_3(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s2_h = _mm256_mul_ps(s2_h, k2_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \
s0_h = _mm256_add_ps(s0_h, s2_h); \

#define kernel_horizontal_liu_4(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s2_h = _mm256_mul_ps(s2_h, k2_h); \
s3_h = _mm256_mul_ps(s3_h, k3_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \
s2_h = _mm256_add_ps(s2_h, s3_h); \
s0_h = _mm256_add_ps(s0_h, s2_h); \

#define kernel_horizontal_liu_5(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s2_h = _mm256_mul_ps(s2_h, k2_h); \
s3_h = _mm256_mul_ps(s3_h, k3_h); \
s4_h = _mm256_mul_ps(s4_h, k4_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \
s2_h = _mm256_add_ps(s2_h, s3_h); \
s0_h = _mm256_add_ps(s0_h, s2_h); \
s0_h = _mm256_add_ps(s0_h, s4_h); \

#define kernel_horizontal_liu_6(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s2_h = _mm256_mul_ps(s2_h, k2_h); \
s3_h = _mm256_mul_ps(s3_h, k3_h); \
s4_h = _mm256_mul_ps(s4_h, k4_h); \
s5_h = _mm256_mul_ps(s5_h, k5_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \
s2_h = _mm256_add_ps(s2_h, s3_h); \
s4_h = _mm256_add_ps(s4_h, s5_h); \
s0_h = _mm256_add_ps(s0_h, s2_h); \
s0_h = _mm256_add_ps(s0_h, s4_h); \

#define kernel_horizontal_liu_7(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s2_h = _mm256_mul_ps(s2_h, k2_h); \
s3_h = _mm256_mul_ps(s3_h, k3_h); \
s4_h = _mm256_mul_ps(s4_h, k4_h); \
s5_h = _mm256_mul_ps(s5_h, k5_h); \
s6_h = _mm256_mul_ps(s6_h, k6_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \
s2_h = _mm256_add_ps(s2_h, s3_h); \
s4_h = _mm256_add_ps(s4_h, s5_h); \
s0_h = _mm256_add_ps(s0_h, s2_h); \
s4_h = _mm256_add_ps(s4_h, s6_h); \
s0_h = _mm256_add_ps(s0_h, s4_h); \

#define kernel_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, k_ptr_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h, k0_h, k1_h, k2_h, k3_h, k4_h, k5_h, k6_h, k7_h) \
load_img_horizontal_liu(src_ptr_h, i_rows, j_cols, pad_size_h, n_col_h, s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
_MM_TRANSPOSE8_PS(s0_h, s1_h, s2_h, s3_h, s4_h, s5_h, s6_h, s7_h); \
s0_h = _mm256_mul_ps(s0_h, k0_h); \
s1_h = _mm256_mul_ps(s1_h, k1_h); \
s2_h = _mm256_mul_ps(s2_h, k2_h); \
s3_h = _mm256_mul_ps(s3_h, k3_h); \
s4_h = _mm256_mul_ps(s4_h, k4_h); \
s5_h = _mm256_mul_ps(s5_h, k5_h); \
s6_h = _mm256_mul_ps(s6_h, k6_h); \
s7_h = _mm256_mul_ps(s7_h, k7_h); \
s0_h = _mm256_add_ps(s0_h, s1_h); \
s2_h = _mm256_add_ps(s2_h, s3_h); \
s4_h = _mm256_add_ps(s4_h, s5_h); \
s6_h = _mm256_add_ps(s6_h, s7_h); \
s0_h = _mm256_add_ps(s0_h, s2_h); \
s4_h = _mm256_add_ps(s4_h, s6_h); \
s0_h = _mm256_add_ps(s0_h, s4_h); \

#define sum_to_d(d0_h, s0_h) \
d0_h = _mm256_add_ps(d0_h, s0_h); \

#define store_d_horizontal_liu(dst_ptr_h, d0_h) \
*(dst_ptr_h + 0) = d0_h[0]; \
*(dst_ptr_h + 1) = d0_h[1]; \
*(dst_ptr_h + 2) = d0_h[2]; \
*(dst_ptr_h + 3) = d0_h[3]; \
*(dst_ptr_h + 4) = d0_h[4]; \
*(dst_ptr_h + 5) = d0_h[5]; \
*(dst_ptr_h + 6) = d0_h[6]; \
*(dst_ptr_h + 7) = d0_h[7]; \


// _mm256_store_ps(dst_ptr_h, d0_h); \
// _mm256_store_ps(dst_ptr_h + 8, d0_h); \

class ParallelHorizontalConvLiu : public cv::ParallelLoopBody {
private:
    int src_row;
    int src_col;
    float* src_ptr_h;
    int dst_row;
    int dst_col;
    float* dst_ptr_h;
    int ksize;
    const float* k_ptr_h;
    const int num_of_simd_for_one_kernel;
    const int partial_SIMD_num;
public:
    ParallelHorizontalConvLiu(int _src_row, int _src_col, float* _src_ptr_h, int _dst_row, int _dst_col, float* _dst_ptr_h, int _ksize, const float* _k_ptr_h, const int _num_of_simd_for_one_kernel, const int _partial_SIMD_num):
    src_row(_src_row), src_col(_src_col), src_ptr_h(_src_ptr_h), dst_row(_dst_row), dst_col(_dst_col), dst_ptr_h(_dst_ptr_h), ksize(_ksize), k_ptr_h(_k_ptr_h), num_of_simd_for_one_kernel(_num_of_simd_for_one_kernel), partial_SIMD_num(_partial_SIMD_num){}
    
    virtual void operator() (const cv::Range& range) const {
        __m256 k0, k1, k2, k3, k4, k5, k6, k7;
        __m256 s0, s1, s2, s3, s4, s5, s6, s7;
        __m256 d0;
        
        int pad_size = (ksize - 1)/2;
        for (int i = range.start; i < range.end; i += 1) {
//            int i = index_transform(i_with_padding, dst_row, pad_size);
            // printf("i:   %i\n\n", i);
            for (int j_ = 0; j_ < src_col; j_ += 8) {
                // printf("j:   %i\n\n", j);
                int j = index_transform(j_, dst_col, pad_size);
                // computational kernel
                d0 = _mm256_setzero_ps();
                // full SIMDs
                int k;
                for (k = 0; k < num_of_simd_for_one_kernel - 1; k++) {
                    load_kernel_horizontal_liu(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                    kernel_horizontal_liu(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                    sum_to_d(d0, s0);
                }
                
                
                // partical SIMDs
                //            printf("k: %i\n",k);
                switch (partial_SIMD_num) {
                    case 1:
                        load_kernel_horizontal_liu_1(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_1(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                    case 2:
                        load_kernel_horizontal_liu_2(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_2(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                    case 3:
                        load_kernel_horizontal_liu_3(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_3(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                    case 4:
                        load_kernel_horizontal_liu_4(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_4(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                    case 5:
                        load_kernel_horizontal_liu_5(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_5(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                        
                    case 6:
                        load_kernel_horizontal_liu_6(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_6(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                        
                    case 7:
                        load_kernel_horizontal_liu_7(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                        kernel_horizontal_liu_7(src_ptr_h, (i+k*8), j, pad_size, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
                        sum_to_d(d0, s0);
                        break;
                        
                    default:
                        break;
                }
                store_d_horizontal_liu(dst_ptr_h+i*dst_col+j_, d0);
            }
        }
        
    }
};

void horizontal_kernel_conv(int src_row, int src_col, float* src_ptr_h, int dst_row, int dst_col, float* dst_ptr_h, int ksize, const float* k_ptr_h) {
    const int num_of_simd_for_one_kernel = (int)ceil(((double)ksize)/8.0);
    //    printf("num_of_simd_for_one_kernel = %i\n", num_of_simd_for_one_kernel);
    
    //    const int half_ksize = ksize/2;
    const int partial_SIMD_num = ksize - 8 * (num_of_simd_for_one_kernel - 1);
    //    printf("partial_SIMD_num = %i\n", partial_SIMD_num);
    
    parallel_for_(cv::Range(0, dst_row), ParallelHorizontalConvLiu(src_row, src_col, src_ptr_h, dst_row, dst_col, dst_ptr_h, ksize, k_ptr_h, num_of_simd_for_one_kernel, partial_SIMD_num));
    
//    for (int i = 0; i < dst_row; i += 1) {
//        // printf("i:   %i\n\n", i);
//        for (int j = 0; j < dst_col; j += 16) {
//            // printf("j:   %i\n\n", j);
//
//            // computational kernel
//            d0 = _mm256_setzero_ps();
//            d1 = _mm256_setzero_ps();
//            // full SIMDs
//            int k;
//            for (k = 0; k < num_of_simd_for_one_kernel - 1; k++) {
//                load_kernel_horizontal_liu(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                kernel_horizontal_liu(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                sum_to_d(d0, s0);
//                kernel_horizontal_liu(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                sum_to_d(d1, s0);
//            }
//
//
//            // partical SIMDs
//            //            printf("k: %i\n",k);
//            switch (partial_SIMD_num) {
//                case 1:
//                    load_kernel_horizontal_liu_1(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_1(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_1(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//                case 2:
//                    load_kernel_horizontal_liu_2(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_2(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_2(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//                case 3:
//                    load_kernel_horizontal_liu_3(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_3(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_3(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//                case 4:
//                    load_kernel_horizontal_liu_4(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_4(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_4(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//                case 5:
//                    load_kernel_horizontal_liu_5(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_5(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_5(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//
//                case 6:
//                    load_kernel_horizontal_liu_6(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_6(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_6(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//
//                case 7:
//                    load_kernel_horizontal_liu_7(k_ptr_h + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_horizontal_liu_7(src_ptr_h+(i+k*8)*dst_col+j, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d0, s0);
//                    kernel_horizontal_liu_7(src_ptr_h+(i+k*8)*dst_col+j+8, src_col, k_ptr_h, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7);
//                    sum_to_d(d1, s0);
//                    break;
//
//                default:
//                    break;
//            }
//            store_d_horizontal_liu(dst_ptr_h+i*dst_col+j, d0, d1);
//        }
//    }
}
