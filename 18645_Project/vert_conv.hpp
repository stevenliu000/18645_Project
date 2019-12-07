//
//  vert_conv.hpp
//  18645_Project
//
//  Created by Steven Liu on 12/6/19.
//  Copyright © 2019 Steven Liu. All rights reserved.
//

#ifndef vert_conv_hpp
#define vert_conv_hpp
#include <iostream>
#include <immintrin.h>
#include <math.h>
#include <string>
#include <fstream>

void vertical_kernel_conv(int src_row, int src_col, float* src_ptr_v, int dst_row, int dst_col, float* dst_ptr_v, int ksize, const float* k_ptr_v);
#endif /* vert_conv_hpp */
