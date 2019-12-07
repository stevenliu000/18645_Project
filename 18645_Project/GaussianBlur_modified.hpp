//
//  GaussianBlur_modified.hpp
//  18645_Project
//
//  Created by Steven Liu on 12/6/19.
//  Copyright © 2019 Steven Liu. All rights reserved.
//

#ifndef GaussianBlur_modified_hpp
#define GaussianBlur_modified_hpp

#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;

void GaussianBlur_modified(InputArray _src, OutputArray _dst, Size ksize,
    double sigma1, double sigma2, int borderType = BORDER_DEFAULT );

#endif /* GaussianBlur_modified_hpp */
