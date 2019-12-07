
#ifndef substraction_kernel_hpp
#define substraction_kernel_hpp

#include <opencv2/core.hpp>
#include <immintrin.h>
using namespace cv;

void substraction_kernel(const Mat& src1, const Mat& src2, Mat& dst);

#endif /* substraction_kernel_hpp */
