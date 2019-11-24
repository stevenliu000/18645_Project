//
//  1a.cpp
//  18645_Project
//
//  Created by Steven Liu on 11/9/19.
//  Copyright © 2019 Steven Liu. All rights reserved.
//

#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <immintrin.h>
#include <math.h>

using namespace std;

#include "1a.hpp"
#include "helper.hpp"
#include "vert_conv.cpp"
#include "hor_conv.cpp"

// float* add_row_padding(float* src, int col_num, int row_num, int ksize) {

//     int pad; float* row_pad;
//     pad = (ksize - 1)/2;
//     row_pad =  new float[row_num*(col_num+2*pad)];

//     cv::copyMakeBorder(src,row_pad,col_num+2*pad,col_num+2*pad,row_num,row_num,4);

//     return row_pad;
// }

// float* add_col_padding(float* src, int col_num, int row_num,int ksize) {

//     int pad; float* col_pad;
//     pad = (ksize - 1)/2;
//     col_pad =  new float[col_num * (row_num + 2 * pad)];

//     cv::copyMakeBorder(src,col_pad,col_num,col_num,row_num+2*pad,row_num+2*pad,4);

//     return col_pad;
// }

/*
 * add_row_padding
 * 
 * input: src - pointer to src
 * 
 * output: pointer to new src with padding (using malloc in function)
 */
float* add_row_padding(float* src, int width, int ksize) {
    int pad, p_width;
    pad = (ksize - 1)/2;
    p_width = width + 2 * pad;
    
    float* row;
    row =  new float[width * p_width + 8];
    
    for (int i = 0; i != width; ++i){
        for (int j = 0; j != p_width; ++j){
            if (j < pad) {
                row[i*p_width+j] = src[i * width + pad - j];
            } else if (j >= (width + pad)) {
                row[i*p_width+j] = src[i * width + pad + 2*(width-1) - j];
            } else {
                row[i*p_width+j] = src[i * width + j - pad];
            }
        }
    }
    return row;
}

/*
 * add_col_padding
 * 
 * input: src - pointer to src
 * 
 * output: pointer to new src with padding (using malloc in function)
 */
float* add_col_padding(float* src, int length,int ksize) {
    int pad, p_length;
    pad = (ksize - 1)/2;
    p_length = length + 2 * pad;
    
    float* col;
    col =  new float[length * p_length];
    
    for (int i = 0; i != p_length; ++i) {
        if (i < pad) {
            for (int j = 0; j != length; ++j) {
                col[i*length+j] = src[(pad-i) * length + j];
            }
        } else if (i >= (length + pad)) {
            for (int j = 0; j != length; ++j) {
                col[i*length+j] = src[(pad + 2*(length-1) - i) * length + j];
            }
        } else {
            for (int j = 0; j != length; ++j) {
                col[i*length+j] = src[(i-pad) * length + j];
            }
        }
    }
    return col;
}

using namespace cv;
Mat getGaussianKernel(int n, double sigma, int ktype)
{
    CV_Assert(n > 0);
    const int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
    small_gaussian_tab[n>>1] : 0;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    Mat kernel(n, 1, ktype);
    float* cf = kernel.ptr<float>();
    double* cd = kernel.ptr<double>();

    double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
    double scale2X = -0.5/(sigmaX*sigmaX);
    double sum = 0;

    int i;
    for( i = 0; i < n; i++ )
    {
        double x = i - (n-1)*0.5;
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
        if( ktype == CV_32F )
        {
            cf[i] = (float)t;
            sum += cf[i];
        }
        else
        {
            cd[i] = t;
            sum += cd[i];
        }
    }

    CV_DbgAssert(fabs(sum) > 0);
    sum = 1./sum;
    for( i = 0; i < n; i++ )
    {
        if( ktype == CV_32F )
            cf[i] = (float)(cf[i]*sum);
        else
            cd[i] *= sum;
    }

    return kernel;
}

template <typename T>
void createGaussianKernels( T & kx, T & ky, int type, Size &ksize,
                                  double sigma1, double sigma2 )
{
    int depth = CV_MAT_DEPTH(type);
    if( sigma2 <= 0 )
        sigma2 = sigma1;

    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        ksize.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;

    CV_Assert( ksize.width  > 0 && ksize.width  % 2 == 1 &&
              ksize.height > 0 && ksize.height % 2 == 1 );

    sigma1 = std::max( sigma1, 0. );
    sigma2 = std::max( sigma2, 0. );

    kx = getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F));
    if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
        ky = kx;
//    else
//        getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F), ky );
}

//Ptr<FilterEngine> createSeparableLinearFilter(
//                                              int _srcType, int _dstType,
//                                              InputArray __rowKernel, InputArray __columnKernel,
//                                              Point _anchor, double _delta,
//                                              int _rowBorderType, int _columnBorderType,
//                                              const Scalar& _borderValue)
//{
//    Mat _rowKernel = __rowKernel.getMat(), _columnKernel = __columnKernel.getMat();
//    _srcType = CV_MAT_TYPE(_srcType);
//    _dstType = CV_MAT_TYPE(_dstType);
//    int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
//    int cn = CV_MAT_CN(_srcType);
//    CV_Assert( cn == CV_MAT_CN(_dstType) );
//    int rsize = _rowKernel.rows + _rowKernel.cols - 1;
//    int csize = _columnKernel.rows + _columnKernel.cols - 1;
//    if( _anchor.x < 0 )
//        _anchor.x = rsize/2;
//    if( _anchor.y < 0 )
//        _anchor.y = csize/2;
//    int rtype = getKernelType(_rowKernel,
//                              _rowKernel.rows == 1 ? Point(_anchor.x, 0) : Point(0, _anchor.x));
//    int ctype = getKernelType(_columnKernel,
//                              _columnKernel.rows == 1 ? Point(_anchor.y, 0) : Point(0, _anchor.y));
//    Mat rowKernel, columnKernel;
//
//    int bdepth = std::max(CV_32F,std::max(sdepth, ddepth));
//    int bits = 0;
//
//    if( sdepth == CV_8U &&
//       ((rtype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
//         ctype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
//         ddepth == CV_8U) ||
//        ((rtype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
//         (ctype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
//         (rtype & ctype & KERNEL_INTEGER) &&
//         ddepth == CV_16S)) )
//    {
//        bdepth = CV_32S;
//        bits = ddepth == CV_8U ? 8 : 0;
//        _rowKernel.convertTo( rowKernel, CV_32S, 1 << bits );
//        _columnKernel.convertTo( columnKernel, CV_32S, 1 << bits );
//        bits *= 2;
//        _delta *= (1 << bits);
//    }
//    else
//    {
//        if( _rowKernel.type() != bdepth )
//            _rowKernel.convertTo( rowKernel, bdepth );
//        else
//            rowKernel = _rowKernel;
//        if( _columnKernel.type() != bdepth )
//            _columnKernel.convertTo( columnKernel, bdepth );
//        else
//            columnKernel = _columnKernel;
//    }
//
//    int _bufType = CV_MAKETYPE(bdepth, cn);
//    Ptr<BaseRowFilter> _rowFilter = getLinearRowFilter(
//                                                       _srcType, _bufType, rowKernel, _anchor.x, rtype);
//    Ptr<BaseColumnFilter> _columnFilter = getLinearColumnFilter(
//                                                                _bufType, _dstType, columnKernel, _anchor.y, ctype, _delta, bits );
//
//    return Ptr<FilterEngine>( new FilterEngine(Ptr<BaseFilter>(), _rowFilter, _columnFilter,
//                                               _srcType, _dstType, _bufType, _rowBorderType, _columnBorderType, _borderValue ));
//}

void conv2d_modified(Mat& _src, Mat& _dst,
                     Mat& kx, Mat& ky, uint64_t &cycles_conv, uint64_t &cycles_mem,
                     int borderType = BORDER_DEFAULT) {
    float *kx_ptr = kx.ptr<float>();
    float *ky_ptr = ky.ptr<float>();
    float *src = _src.ptr<float>();
    float *dst = _dst.ptr<float>();
    
    uint64_t start, end;

    int k_len = max(kx.cols, kx.rows);

    /*  /----/
     *  /----/
     */
    // start = rdtsc();
    float *src_row_padding = add_row_padding(src, static_cast<int>(_dst.cols), k_len);
    // end = rdtsc();
    // cycles_mem += end-start;
    // start = rdtsc();
    horizontal_kernel_conv(_src.rows, _src.cols+k_len-1, src_row_padding, _dst.rows, _dst.cols, dst, k_len, kx_ptr);
    // end = rdtsc();
    // cycles_conv += end-start;
    /*  |-|
     *  | |
     *  |-|
     */
    start = rdtsc();
    float *src_col_padding = add_col_padding(dst, static_cast<int>(_dst.rows), k_len);
    end = rdtsc();
    cycles_mem += end-start;
    start = rdtsc();
    vertical_kernel_conv(_src.rows+k_len-1, _src.cols, src_col_padding, _dst.rows, _dst.cols, dst, k_len, ky_ptr);
    end = rdtsc();
    cycles_conv += end-start;
//    delete src_row_padding;
//    delete src_col_padding;
}

/*
 * sepFilter2D_modified: call conv2d_modified
 */
void sepFilter2D_modified(Mat& src, Mat& dst, Mat& kx, Mat& ky, 
                          int borderType, uint64_t &cycles_conv, uint64_t &cycles_mem) {
                          
                          
//                          int stype, int dtype, int ktype,
//                          uchar* src_data, size_t src_step, uchar* dst_data, size_t dst_step,
//                          int width, int height, int full_width, int full_height,
//                          int offset_x, int offset_y,
//                          uchar * kernelx_data, int kernelx_len,
//                          uchar * kernely_data, int kernely_len,
//                          int anchor_x, int anchor_y, double delta, int borderType) {
//    Mat kernelX(Size(kernelx_len, 1), ktype, kernelx_data);
//    Mat kernelY(Size(kernely_len, 1), ktype, kernely_data);
//    Ptr<FilterEngine> f = createSeparableLinearFilter(stype, dtype, kernelX, kernelY,
//                                                      Point(anchor_x, anchor_y),
//                                                      delta, borderType & ~BORDER_ISOLATED);
//    Mat src(Size(width, height), stype, src_data, src_step);
//    Mat dst(Size(width, height), dtype, dst_data, dst_step);
    conv2d_modified(src, dst, kx, ky, cycles_conv, cycles_mem);
//    f->apply(src, dst, Size(full_width, full_height), Point(offset_x, offset_y));
}


void GaussianBlur_modified(InputArray _src, OutputArray _dst, Size ksize,
                           double sigma1, double sigma2, uint64_t &cycles_conv, uint64_t &cycles_mem, 
                       int borderType = BORDER_DEFAULT ) {
    int type = _src.type();
    Size size = _src.size();
    _dst.create( size, type );

    if( (borderType & ~BORDER_ISOLATED) != BORDER_CONSTANT &&
       ((borderType & BORDER_ISOLATED) != 0 || !_src.getMat().isSubmatrix()) )
    {
        if( size.height == 1 )
            ksize.height = 1;
        if( size.width == 1 )
            ksize.width = 1;
    }

    if( ksize.width == 1 && ksize.height == 1 )
    {
        _src.copyTo(_dst);
        return;
    }

    int sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType & BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

//    sepFilter2D_modified(src, dst, sdepth, kx, ky, Point(-1, -1), 0, borderType);
//
    sepFilter2D_modified(src, dst, kx, ky, borderType, cycles_conv, cycles_mem);
}
