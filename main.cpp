//
//  main.cpp
//  Demo
//
//  Created by tonye on 2018/12/3.
//  Copyright © 2018 tonye. All rights reserved.
//


#include <opencv2/opencv.hpp>
#include "lib/OpenCVOp/OpencvTest.h"
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, const char * argv[]) {
    // insert code here...

    int res = 0;

    OpencvTest* opencvTest = new OpencvTest();
    string imgPath = "/Users/tonye/CLionProjects/CLionProjects-Opencv/OpenCVApp/resources/timg.jpg";

    //res = opencvTest->getRGBscaleImage(imgPath);
    //res = opencvTest->drawCircleImage(imgPath);
    //res = opencvTest->maskImage(imgPath);
    
    //res = opencvTest->matOp(imgPath);
//    string imgpath_pattern = "/Users/tonye/Downloads/智能安检-压力罐样本1204/罐子4/*.png";
//    res = opencvTest->readImgFiles(imgpath_pattern, "/Users/tonye/Downloads/004");
    
    //res = opencvTest->imageOp(imgPath);
    
    //res = opencvTest->imageMixture(imgPath, imgPath);
    
    //res = opencvTest->adjustImageLuminanceL(imgPath);
    
    //res = opencvTest->drawText(imgPath);
    
    //res = opencvTest->blurImage(imgPath);

    //腐蚀和膨胀
    //res = opencvTest->corrosionAndSwell(imgPath);

    //形态学操作
    std::string imgPath2 = "/Users/tonye/CLionProjects/CLionProjects-OpenCV/OpenCVApp/resources/zaodian.jpg";
    res = opencvTest->morphologyOp(imgPath2);

    //形态学操作应用-提取水平与垂直线

    return res;
    
    
}
