
#include "GaussianBlur_modified.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <immintrin.h>
#include <math.h>
#include "helper.hpp"
#include "vert_conv.hpp"

using namespace std;
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
}

void conv2d_modified(Mat& _src, Mat& _dst,
                     Mat& kx, Mat& ky, int borderType = BORDER_DEFAULT) {
    float *kx_ptr = kx.ptr<float>();
    float *ky_ptr = ky.ptr<float>();

    float *src = _src.ptr<float>();
    float *dst = _dst.ptr<float>();
    
    int k_len = max(kx.cols, kx.rows);

    Mat _tmp(_src.rows, _src.cols, CV_32F);
    float *tmp = _tmp.ptr<float>();

    vertical_kernel_conv(_src.rows+k_len-1, _src.cols, src, _dst.rows, _dst.cols, tmp, k_len, ky_ptr);
    vertical_kernel_conv(_src.rows+k_len-1, _src.cols, tmp, _dst.rows, _dst.cols, dst, k_len, ky_ptr);
}

/*
 * sepFilter2D_modified: call conv2d_modified
 */
void sepFilter2D_modified(Mat& src, Mat& dst, Mat& kx, Mat& ky, int borderType) {
    conv2d_modified(src, dst, kx, ky);
}


void GaussianBlur_modified(InputArray _src, OutputArray _dst, Size ksize,
                           double sigma1, double sigma2, int borderType) {
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

    sepFilter2D_modified(src, dst, kx, ky, borderType);
}
