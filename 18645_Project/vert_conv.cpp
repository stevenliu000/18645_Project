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


// inline int index_transform(int index, int num_col, int pad_size) {
//     int new_index;
//     if (index < pad_size)
//         new_index = pad_size - index;
//     else if (index >= num_col + pad_size)
//         new_index = num_col - 2 - (index - num_col - pad_size);
//     else
//         new_index = index - pad_size;
    
//     return new_index;
// }


#define load_kernel_vertical_1(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \

#define load_kernel_vertical_2(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \

#define load_kernel_vertical_3(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \
k2_v = _mm256_set1_ps(*(k_ptr_v + 2)); \

#define load_kernel_vertical_4(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \
k2_v = _mm256_set1_ps(*(k_ptr_v + 2)); \
k3_v = _mm256_set1_ps(*(k_ptr_v + 3)); \

#define load_kernel_vertical_5(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \
k2_v = _mm256_set1_ps(*(k_ptr_v + 2)); \
k3_v = _mm256_set1_ps(*(k_ptr_v + 3)); \
k4_v = _mm256_set1_ps(*(k_ptr_v + 4)); \

#define load_kernel_vertical_6(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \
k2_v = _mm256_set1_ps(*(k_ptr_v + 2)); \
k3_v = _mm256_set1_ps(*(k_ptr_v + 3)); \
k4_v = _mm256_set1_ps(*(k_ptr_v + 4)); \
k5_v = _mm256_set1_ps(*(k_ptr_v + 5)); \

#define load_kernel_vertical_7(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \
k2_v = _mm256_set1_ps(*(k_ptr_v + 2)); \
k3_v = _mm256_set1_ps(*(k_ptr_v + 3)); \
k4_v = _mm256_set1_ps(*(k_ptr_v + 4)); \
k5_v = _mm256_set1_ps(*(k_ptr_v + 5)); \
k6_v = _mm256_set1_ps(*(k_ptr_v + 6)); \

#define load_kernel_vertical(k_ptr_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v) \
k0_v = _mm256_set1_ps(*(k_ptr_v + 0)); \
k1_v = _mm256_set1_ps(*(k_ptr_v + 1)); \
k2_v = _mm256_set1_ps(*(k_ptr_v + 2)); \
k3_v = _mm256_set1_ps(*(k_ptr_v + 3)); \
k4_v = _mm256_set1_ps(*(k_ptr_v + 4)); \
k5_v = _mm256_set1_ps(*(k_ptr_v + 5)); \
k6_v = _mm256_set1_ps(*(k_ptr_v + 6)); \
k7_v = _mm256_set1_ps(*(k_ptr_v + 7)); \

#define load_img_vertical_1(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical_2(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical_3(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \
s2_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 2, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical_4(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \
s2_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 2, n_col_v, pad_size_v) * n_col_v + j); \
s3_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 3, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical_5(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \
s2_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 2, n_col_v, pad_size_v) * n_col_v + j); \
s3_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 3, n_col_v, pad_size_v) * n_col_v + j); \
s4_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 4, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical_6(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \
s2_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 2, n_col_v, pad_size_v) * n_col_v + j); \
s3_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 3, n_col_v, pad_size_v) * n_col_v + j); \
s4_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 4, n_col_v, pad_size_v) * n_col_v + j); \
s5_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 5, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical_7(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \
s2_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 2, n_col_v, pad_size_v) * n_col_v + j); \
s3_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 3, n_col_v, pad_size_v) * n_col_v + j); \
s4_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 4, n_col_v, pad_size_v) * n_col_v + j); \
s5_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 5, n_col_v, pad_size_v) * n_col_v + j); \
s6_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 6, n_col_v, pad_size_v) * n_col_v + j); \

#define load_img_vertical(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v) \
s0_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 0, n_col_v, pad_size_v) * n_col_v + j); \
s1_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 1, n_col_v, pad_size_v) * n_col_v + j); \
s2_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 2, n_col_v, pad_size_v) * n_col_v + j); \
s3_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 3, n_col_v, pad_size_v) * n_col_v + j); \
s4_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 4, n_col_v, pad_size_v) * n_col_v + j); \
s5_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 5, n_col_v, pad_size_v) * n_col_v + j); \
s6_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 6, n_col_v, pad_size_v) * n_col_v + j); \
s7_v = _mm256_load_ps(src_ptr_v + index_transform(i_rows + 7, n_col_v, pad_size_v) * n_col_v + j); \

#define kernel_veritcal_1(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_1(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \

#define kernel_veritcal_2(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_2(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \

#define kernel_veritcal_3(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_3(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \
d0_v = _mm256_fmadd_ps(s2_v, k2_v, d0_v); \

#define kernel_veritcal_4(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_4(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \
d0_v = _mm256_fmadd_ps(s2_v, k2_v, d0_v); \
d0_v = _mm256_fmadd_ps(s3_v, k3_v, d0_v); \

#define kernel_veritcal_5(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_5(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \
d0_v = _mm256_fmadd_ps(s2_v, k2_v, d0_v); \
d0_v = _mm256_fmadd_ps(s3_v, k3_v, d0_v); \
d0_v = _mm256_fmadd_ps(s4_v, k4_v, d0_v); \

#define kernel_veritcal_6(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_6(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \
d0_v = _mm256_fmadd_ps(s2_v, k2_v, d0_v); \
d0_v = _mm256_fmadd_ps(s3_v, k3_v, d0_v); \
d0_v = _mm256_fmadd_ps(s4_v, k4_v, d0_v); \
d0_v = _mm256_fmadd_ps(s5_v, k5_v, d0_v); \

#define kernel_veritcal_7(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical_7(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \
d0_v = _mm256_fmadd_ps(s2_v, k2_v, d0_v); \
d0_v = _mm256_fmadd_ps(s3_v, k3_v, d0_v); \
d0_v = _mm256_fmadd_ps(s4_v, k4_v, d0_v); \
d0_v = _mm256_fmadd_ps(s5_v, k5_v, d0_v); \
d0_v = _mm256_fmadd_ps(s6_v, k6_v, d0_v); \

#define kernel_veritcal(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, k_ptr_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v, k0_v, k1_v, k2_v, k3_v, k4_v, k5_v, k6_v, k7_v, d0_v) \
load_img_vertical(src_ptr_v, i_rows, j_cols, pad_size_v, n_col_v, s0_v, s1_v, s2_v, s3_v, s4_v, s5_v, s6_v, s7_v); \
d0_v = _mm256_fmadd_ps(s0_v, k0_v, d0_v); \
d0_v = _mm256_fmadd_ps(s1_v, k1_v, d0_v); \
d0_v = _mm256_fmadd_ps(s2_v, k2_v, d0_v); \
d0_v = _mm256_fmadd_ps(s3_v, k3_v, d0_v); \
d0_v = _mm256_fmadd_ps(s4_v, k4_v, d0_v); \
d0_v = _mm256_fmadd_ps(s5_v, k5_v, d0_v); \
d0_v = _mm256_fmadd_ps(s6_v, k6_v, d0_v); \
d0_v = _mm256_fmadd_ps(s7_v, k7_v, d0_v); \

#define sum_to_d(d0_v, s0_v) \
d0_v = _mm256_add_ps(d0_v, s0_v); \

#define store_d_vertical(dst_ptr_v, n_col_v, d0_v, d1_v) \
*(dst_ptr_v + 0 * n_col_v) = d0_v[0]; \
*(dst_ptr_v + 1 * n_col_v) = d0_v[1]; \
*(dst_ptr_v + 2 * n_col_v) = d0_v[2]; \
*(dst_ptr_v + 3 * n_col_v) = d0_v[3]; \
*(dst_ptr_v + 4 * n_col_v) = d0_v[4]; \
*(dst_ptr_v + 5 * n_col_v) = d0_v[5]; \
*(dst_ptr_v + 6 * n_col_v) = d0_v[6]; \
*(dst_ptr_v + 7 * n_col_v) = d0_v[7]; \
*(dst_ptr_v + 8 * n_col_v) = d1_v[0]; \
*(dst_ptr_v + 9 * n_col_v) = d1_v[1]; \
*(dst_ptr_v + 10 * n_col_v) = d1_v[2]; \
*(dst_ptr_v + 11 * n_col_v) = d1_v[3]; \
*(dst_ptr_v + 12 * n_col_v) = d1_v[4]; \
*(dst_ptr_v + 13 * n_col_v) = d1_v[5]; \
*(dst_ptr_v + 14 * n_col_v) = d1_v[6]; \
*(dst_ptr_v + 15 * n_col_v) = d1_v[7]; \

//*(dst_ptr_v + 0) = d0_v[0]; \
//*(dst_ptr_v + 1) = d0_v[1]; \
//*(dst_ptr_v + 2) = d0_v[2]; \
//*(dst_ptr_v + 3) = d0_v[3]; \
//*(dst_ptr_v + 4) = d0_v[4]; \
//*(dst_ptr_v + 5) = d0_v[5]; \
//*(dst_ptr_v + 6) = d0_v[6]; \
//*(dst_ptr_v + 7) = d0_v[7]; \
//*(dst_ptr_v + 8) = d1_v[0]; \
//*(dst_ptr_v + 9) = d1_v[1]; \
//*(dst_ptr_v + 10) = d1_v[2]; \
//*(dst_ptr_v + 11) = d1_v[3]; \
//*(dst_ptr_v + 12) = d1_v[4]; \
//*(dst_ptr_v + 13) = d1_v[5]; \
//*(dst_ptr_v + 14) = d1_v[6]; \
//*(dst_ptr_v + 15) = d1_v[7]; \


// _mm256_store_ps(dst_ptr_v, d0_v); \
// _mm256_store_ps(dst_ptr_v + 8, d0_v); \

//class ParallelVerticalConv : public cv::ParallelLoopBody {
//private:
//    int src_row;
//    int src_col;
//    float* src_ptr_v;
//    int dst_row;
//    int dst_col;
//    float* dst_ptr_v;
//    int ksize;
//    const float* k_ptr_v;
//    const int num_of_simd_for_one_kernel;
//    const int partial_SIMD_num;
//public:
//    ParallelVerticalConv(int _src_row, int _src_col, float* _src_ptr_v, int _dst_row, int _dst_col, float* _dst_ptr_v, int _ksize, const float* _k_ptr_v, const int _num_of_simd_for_one_kernel, const int _partial_SIMD_num):
//    src_row(_src_row), src_col(_src_col), src_ptr_v(_src_ptr_v), dst_row(_dst_row), dst_col(_dst_col), dst_ptr_v(_dst_ptr_v), ksize(_ksize), k_ptr_v(_k_ptr_v), num_of_simd_for_one_kernel(_num_of_simd_for_one_kernel), partial_SIMD_num(_partial_SIMD_num){}
//
//    virtual void operator() (const cv::Range& range) const {
//        __m256 k0, k1, k2, k3, k4, k5, k6, k7;
//        __m256 s0, s1, s2, s3, s4, s5, s6, s7;
//        __m256 d0, d1;
//        int j, i, j_, i_with_padding;
//        int pad_size = (ksize - 1)/2;
//            // printf("i:   %i\n\n", i);
////                        j = j_*16;
//
//        for (i_with_padding = range.start; i_with_padding < range.end; i_with_padding += 1) {
//            i = index_transform(i_with_padding, dst_row, pad_size);
//            for (j = 0; j < dst_col; j += 16) {
//
//
//                // printf("j:   %i\n\n", j);
//
//                // computational kernel
//                d0 = _mm256_setzero_ps();
//                d1 = _mm256_setzero_ps();
//                // full SIMDs
//                int k;
//                for (k = 0; k < num_of_simd_for_one_kernel - 1; k++) {
//                    load_kernel_vertical_4(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                    kernel_veritcal_4(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                    kernel_veritcal_4(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                }
//
//
//                // partical SIMDs
//                //            printf("k: %i\n",k);
//                switch (partial_SIMD_num) {
//                    case 1:
//                        load_kernel_vertical_1(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_1(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_1(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//                    case 2:
//                        load_kernel_vertical_2(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_2(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_2(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//                    case 3:
//                        load_kernel_vertical_3(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_3(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_3(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//                    case 4:
//                        load_kernel_vertical_4(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_4(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_4(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//                    case 5:
//                        load_kernel_vertical_5(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_5(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_5(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//
//                    case 6:
//                        load_kernel_vertical_6(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_6(src_ptr_v, (i+k*8), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_6(src_ptr_v, (i+k*8), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//
//                    case 7:
//                        load_kernel_vertical_7(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
//                        kernel_veritcal_7(src_ptr_v, (i+k*8), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
//                        kernel_veritcal_7(src_ptr_v, (i+k*8), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
//                        break;
//
//                    default:
//                        break;
//                }
//                store_d_vertical(dst_ptr_v+j*dst_col+i_with_padding, dst_col, d0, d1);
//            }
//        }
//
//    }
//};

void vertical_kernel_conv(int src_row, int src_col, float* src_ptr_v, int dst_row, int dst_col, float* dst_ptr_v, int ksize, const float* k_ptr_v) {
    const int num_of_kernel_for_one_filter = (int)ceil(((double)ksize)/4.0);
    //    printf("num_of_simd_for_one_kernel = %i\n", num_of_simd_for_one_kernel);
    
    //    const int half_ksize = ksize/2;
    const int partial_kernel_num = ksize - 4 * (num_of_kernel_for_one_filter - 1);
    //    printf("partial_SIMD_num = %i\n", partial_SIMD_num);
    
//    parallel_for_(cv::Range(0, dst_row), ParallelVerticalConv(src_row, src_col, src_ptr_v, dst_row, dst_col, dst_ptr_v, ksize, k_ptr_v, num_of_kernel_for_one_filter, partial_kernel_num));

        __m256 k0, k1, k2, k3, k4, k5, k6, k7;
        __m256 s0, s1, s2, s3, s4, s5, s6, s7;
        __m256 d0, d1;
        int pad_size = (ksize - 1)/2;
            // printf("i:   %i\n\n", i);
        for (int i_with_padding = 0; i_with_padding < dst_row; i_with_padding += 1) {
            int i = index_transform(i_with_padding, dst_row, pad_size);

        for (int j = 0; j < dst_col; j += 16) {
//            j = j_*16;

                // printf("j:   %i\n\n", j);

                // computational kernel
                d0 = _mm256_setzero_ps();
                d1 = _mm256_setzero_ps();
                // full SIMDs
                int k;
                                for (k = 0; k < num_of_kernel_for_one_filter - 1; k++) {
                                    load_kernel_vertical_4(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                    kernel_veritcal_4(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                    kernel_veritcal_4(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                }
                
                
                                // partical SIMDs
                                //            printf("k: %i\n",k);
                                switch (partial_kernel_num) {
                                    case 1:
                                        load_kernel_vertical_1(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_1(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_1(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                                    case 2:
                                        load_kernel_vertical_2(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_2(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_2(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                                    case 3:
                                        load_kernel_vertical_3(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_3(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_3(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                                    case 4:
                                        load_kernel_vertical_4(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_4(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_4(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                                    case 5:
                                        load_kernel_vertical_5(k_ptr_v + 4 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_5(src_ptr_v, (i+k*4), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_5(src_ptr_v, (i+k*4), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                
                                    case 6:
                                        load_kernel_vertical_6(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_6(src_ptr_v, (i+k*8), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_6(src_ptr_v, (i+k*8), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                
                                    case 7:
                                        load_kernel_vertical_7(k_ptr_v + 8 * k, k0, k1, k2, k3, k4, k5, k6, k7);
                                        kernel_veritcal_7(src_ptr_v, (i+k*8), j, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d0);
                                        kernel_veritcal_7(src_ptr_v, (i+k*8), j+8, pad_size, src_col, k_ptr_v, s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, k4, k5, k6, k7, d1);
                                        break;
                
                                    default:
                                        break;
                                }
                                store_d_vertical(dst_ptr_v+j*dst_col+i_with_padding, dst_col, d0, d1);
                            }
                        }
                
                    
}
