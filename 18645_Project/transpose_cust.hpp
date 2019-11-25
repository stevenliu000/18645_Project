// #define _MM_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
// 	do { \
// 		__m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7; \
// 		__m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7; \
// 		__t0 = _mm256_unpacklo_ps(row0, row1); \
// 		__t1 = _mm256_unpackhi_ps(row0, row1); \
// 		__t2 = _mm256_unpacklo_ps(row2, row3); \
// 		__t3 = _mm256_unpackhi_ps(row2, row3); \
// 		__t4 = _mm256_unpacklo_ps(row4, row5); \
// 		__t5 = _mm256_unpackhi_ps(row4, row5); \
// 		__t6 = _mm256_unpacklo_ps(row6, row7); \
// 		__t7 = _mm256_unpackhi_ps(row6, row7); \
// 		__tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0)); \
// 		__tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2)); \
// 		__tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0)); \
// 		__tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2)); \
// 		__tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0)); \
// 		__tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2)); \
// 		__tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0)); \
// 		__tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2)); \
// 		row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20); \
// 		row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20); \
// 		row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20); \
// 		row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20); \
// 		row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31); \
// 		row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31); \
// 		row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31); \
// 		row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31); \
// 	} while (0)

void inline transpose_iterative(float *a, float *b, int N) {
 int row, col;

 for (row = 0; row < N; row++) {
   for (col = 0; col < N; col++) {
     *(b+col*N+row) = *(a+row*N+col);
   }
 }
}

void transpose_cust(float *a, int lda, float *b, int ldb, int N) {
 if (N<=128){
   transpose_iterative(a, b, N);
 }
 else{
   //Top-Left quadrant
   transpose_cust(a,               lda, b, ldb, N/2);

   //Bottom-Left quadrant
   transpose_cust(a+N/2,           lda, b+(N/2*ldb), ldb, N/2);

   //Bottom-Right quadrant
   transpose_cust(a+(N/2*lda+N/2), lda, b+(N/2*lda+N/2), ldb, N/2);

   //Top-Right quadrant
   transpose_cust(a+(N/2*lda),     lda, b+N/2, ldb, N/2);
 } 
}
