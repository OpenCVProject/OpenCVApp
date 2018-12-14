//
//  Base.cpp
//  Demo
//
//  Created by tonye on 2018/12/3.
//  Copyright © 2018 tonye. All rights reserved.
//


#include "Base.h"
#include <opencv2/opencv.hpp>
#include <iostream>


bool Base::checkImgData(std::string img)
{
    cv::Mat imgMat = cv::imread(img);
    bool flag = imgMat.data?true:false;
    std::cout <<  "check img " << img << " " << flag << std::endl;
    return flag;
}
