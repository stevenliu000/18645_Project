#define kernel(src1, src2, dst, src1a, src1b, src1c, src1d, src2a, src2b, src2c, src2d, dest1, dest2, dest3, dest4) \
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