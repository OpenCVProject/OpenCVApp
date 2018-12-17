//
//  OpencvTest.hpp
//  Demo
//
//  Created by tonye on 2018/12/3.
//  Copyright © 2018 tonye. All rights reserved.
//
#include <iostream>
#include "Base.h"
# pragma once
class OpencvTest: public Base{
    public:
        int getRGBscaleImage(std::string img);
        int drawCircleImage(std::string img);
        int maskImage(std::string img);
        int matOp(std::string img);
        int readImgFiles(std::string path, std::string output);
        int imageOp(std::string path);
        int imageMixture(std::string path1,std::string path2);
        int adjustImageLuminanceL(std::string path);
        int drawText(std::string path);
        int blurImage(std::string path);
        /**
         * 腐蚀和膨胀
         * @param path
         * @return
         */
        int corrosionAndSwell(std::string path);

        /**
         * 形态学操作
         * @param paht
         * @return
         */
        int morphologyOp(std::string paht);


};

