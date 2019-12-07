#include "substraction_kernel.hpp"
#include <opencv2/core.hpp>
#include <immintrin.h>

using namespace cv;

#define kernel(src1_ptr, src2_ptr, dst, src1a, src1b, src1c, src1d, src2a, src2b, src2c, src2d, dest1, dest2, dest3, dest4) \
src1a = _mm256_load_ps(&src1_ptr[k]); \
src1b = _mm256_load_ps(&src1_ptr[k + 8]); \
src1c = _mm256_load_ps(&src1_ptr[k + 16]); \
src1d = _mm256_load_ps(&src1_ptr[k + 24]); \
src2a = _mm256_load_ps(&src2_ptr[k]); \
src2b = _mm256_load_ps(&src2_ptr[k + 8]); \
src2c = _mm256_load_ps(&src2_ptr[k + 16]); \
src2d = _mm256_load_ps(&src2_ptr[k + 24]); \
dest1 = _mm256_sub_ps(src2a, src1a); \
dest2 = _mm256_sub_ps(src2b, src1b); \
dest3 = _mm256_sub_ps(src2c, src1c); \
dest4 = _mm256_sub_ps(src2d, src1d); \
_mm256_store_ps(&dst_ptr[k], dest1); \
_mm256_store_ps(&dst_ptr[k + 8], dest2); \
_mm256_store_ps(&dst_ptr[k + 16], dest3); \

void substraction_kernel(const Mat& src1, const Mat& src2, Mat& dst) {

  dst.create(src2.rows, src2.cols, CV_32F);

  const float *src1_ptr = src1.ptr<float>();
  const float *src2_ptr = src2.ptr<float>();
  float *dst_ptr = dst.ptr<float>();
    
  __m256 src1a, src1b, src1c, src1d;
  __m256 src2a, src2b, src2c, src2d;
  __m256 dest1, dest2, dest3, dest4;

  for (int k = 0; k <= src2.rows * src2.cols - 32; k += 32) {
    kernel(src1_ptr, src2_ptr, dst, src1a, src1b, src1c, src1d, src2a, src2b, src2c, src2d, dest1, dest2, dest3, dest4);
    }
}