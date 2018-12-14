//
//  OpencvTest.cpp
//  Demo
//
//  Created by tonye on 2018/12/3.
//  Copyright © 2018 tonye. All rights reserved.
//

#include "OpencvTest.h"
#include <opencv2/opencv.hpp>



int OpencvTest::getRGBscaleImage(std::string img)
{
    if(!checkImgData(img))
    {
        std::cout << "image check fail!!!";
        return 1;
    }else{
        cv::Mat m = cv::imread(img);
        //int *p_address;
        cv::Vec3i color;
        for (int col = 20; col < 40; col++)
        {
            for (int row = 2; row < 20; row++)
            {
                color[0] = (int)(*(m.data + m.step[0] * row + m.step[1] * col));
                color[1] = (int)(*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1()));
                color[2] = (int)(*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1() * 2));
                
                std::cout << color[0] << "," << color[1] << "," << color[2] << " ==>";
                
                color[0] = 255;
                color[1] = 0;
                color[2] = 0;
                
                *(m.data + m.step[0] * row + m.step[1] * col) = color[0];
                *(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1()) = color[1];
                *(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1() * 2) = color[2];
                
                std::cout << (int)*(m.data + m.step[0] * row + m.step[1] * col) << "," << (int)*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1()) << "," << (int)*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1() * 2) << std::endl;
            
            }
        }
        
        cv::imshow("ImageShow", m);
        cv::waitKey();
      
    }
    return 0;
}

int OpencvTest::drawCircleImage(std::string img)
{
    if(!checkImgData(img))
    {
        std::cout << "image check fail!!!";
        return 1;
    }else{
        cv::Mat m = cv::imread(img);
        cv::namedWindow("org", CV_WINDOW_AUTOSIZE);
        cv::imshow("org Show", m);
        
        cv::Mat output_image;
        cv::cvtColor(m, output_image, CV_BGR2HLS);
        cv::namedWindow("out", CV_WINDOW_AUTOSIZE);
        cv::imshow("out Show", output_image);
        
        cv::imwrite("/Users/tonye/Downloads/test.tif", output_image);

        
        cv::Mat_<cv::Vec3b> m2 = m;
        
        for (int y = 21; y < 42; y++)
        {
            for (int x = 2; x < 21; x++)
            {
                if(std::pow(double(x - 11), 2) + std::pow(double(y - 31), 2) - 64.0 < 0.0000)
                {
                    m2(x, y) = cv::Vec3b(0, 0, 255);
                }
            }
        }
        
        cv::namedWindow("process", CV_WINDOW_AUTOSIZE);
        cv::imshow("process Show", m);
        
        cv::waitKey();
    }
    return 0;
}

/**
 * 图像的掩膜操作
 */
int OpencvTest::maskImage(std::string img)
{
    cv::Mat src, dst;
    src = cv::imread(img);
    if (src.empty()){
        std::printf("could not load image ...\n");
    }
    
    cv::namedWindow("input image", CV_WINDOW_AUTOSIZE);
    cv::imshow("input image", src);
    
    //手工定义掩膜
//    double t = (double)cv::getTickCount();
//    int cols = lib.cols * lib.channels();  //宽度
//    int rows = lib.rows;  //高度
//    int offsets = lib.channels();
//
//    dst = cv::Mat::zeros(lib.size(), lib.type());
//    for(int row = 1; row < rows - 1; row ++){
//        //获取图像像素指针
//        const uchar* previous = lib.ptr<uchar>(row-1);
//        const uchar* current = lib.ptr<uchar>(row);
//        const uchar* next = lib.ptr<uchar>(row+1);
//
//        uchar* output = dst.ptr<uchar>(row);
//        for(int col = offsets; col < cols; col++){
//            output[col] = cv::saturate_cast<uchar>(5*current[col] - (current[col-offsets]+current[col+offsets]+previous[col]+next[col]));
//        }
//    }
//    double timeconsume = (cv::getTickCount()-t)/cv::getTickFrequency();
//    std::printf("time consume %.5f \n", timeconsume);
    
    //调用接口定义掩膜
    double t2 = (double)cv::getTickCount();
    cv::Mat kernel = (cv::Mat_<char>(3,3) << 0,-1,0,-1,5,-1,0,-1,0);
    cv::filter2D(src, dst, src.depth(), kernel);
    double timeconsume2 = (cv::getTickCount()-t2)/cv::getTickFrequency();
    std::printf("time consume2 %.5f \n", timeconsume2);
    
    cv::namedWindow("constrast image demo", CV_WINDOW_AUTOSIZE);
    cv::imshow("constrast image demo", dst);
    
    cv::waitKey();
    
    return 0;
    
}


/**
 * Mat 操作
 */
int OpencvTest::matOp(std::string img){
    cv::Mat src;
    src = cv::imread(img);
    if(src.empty()){
        std::cout << "could not load  image ..." << std::endl;
        return -1;
    }
    cv::namedWindow("input", CV_WINDOW_AUTOSIZE);
    cv::imshow("input", src);
    
//    cv::Mat dst;
//    dst = cv::Mat(lib.size(),lib.type());
//    dst = cv::Scalar(127, 0, 255);
//
//    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
//    cv::imshow("output", dst);
    
    cv::Mat dst = src.clone();

    
    cv::cvtColor(src, dst, CV_BGR2GRAY);
    std::printf("input channels %d \n", src.channels());
    std::printf("input channels %d \n", dst.channels());
    
    int cols = dst.cols;
    int rows = dst.rows;
    int channels = dst.channels();
    std::printf("rows: %d cols: %d channels: %d\n", rows, cols, channels);

    const uchar* firstRow = dst.ptr<uchar>(0);
    std::printf("first pixel value: %d", *firstRow);
    
    
    
   /*
    //用构造函数创建Mat
    cv::Mat m(100, 100, CV_8UC3, cv::Scalar(0,0,255));
    std::cout << 'm = '<< std::endl << m << std::endl;
    
    //深拷贝和浅拷贝
    cv::Mat A_shallowCopy(m);
    cv::Mat B_deepCopy;
    //B_deepCopy = m.clone();
    m.copyTo(B_deepCopy);

    
    cv::Vec3i color;
    for (int col = 20; col < 40; col++)
    {
        for (int row = 2; row < 20; row++)
        {
            color[0] = (int)(*(m.data + m.step[0] * row + m.step[1] * col));
            color[1] = (int)(*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1()));
            color[2] = (int)(*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1() * 2));
            
            std::cout << color[0] << "," << color[1] << "," << color[2] << " ==>";
            
            color[0] = 255;
            color[1] = 0;
            color[2] = 0;
            
            *(m.data + m.step[0] * row + m.step[1] * col) = color[0];
            *(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1()) = color[1];
            *(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1() * 2) = color[2];
            
            std::cout << (int)*(m.data + m.step[0] * row + m.step[1] * col) << "," << (int)*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1()) << "," << (int)*(m.data + m.step[0] * row + m.step[1] * col + m.elemSize1() * 2) << std::endl;
            
        }
    }
    
    
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", A_shallowCopy);
    cv::namedWindow("output2", CV_WINDOW_AUTOSIZE);
    cv::imshow("output2", B_deepCopy);
    */
    
    
    /*
    //用create创建Mat
    cv::Mat m1;
    m1.create(lib.size(), lib.type());
    m1 = cv::Scalar(0, 0, 255);
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", m1);
     */
    
    
    /*
    //定义小数组  掩膜 图片增强
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(lib, dst, lib.depth(), kernel);
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", dst);
     */
    
    /*
    //Matlabg风格创建Mat Mat::zeros  Mat::eye
    cv::Mat m2 = cv::Mat::zeros(lib.size(), lib.type());
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", m2);
    
    cv::Mat m3 = cv::Mat::zeros(2, 2, CV_8UC3);
    std::cout << 'm3 = '<< std::endl << m3 << std::endl;
    cv::Mat m4 = cv::Mat::eye(2, 2, CV_8UC3);
    std::cout << 'm3 = '<< std::endl << m4 << std::endl;
    */
    
    
    
    cv::waitKey();
    return 0;
}

/**
 * 批量读取图片
 */
int OpencvTest::readImgFiles(std::string imgpath_pattern, std::string output){
    
    std::vector<cv::String> image_files;
    cv::glob(imgpath_pattern, image_files);
    if (image_files.size() == 0) {
        std::cout << imgpath_pattern << std::endl;
        std::cout << "No image files " << std::endl;
        return 0;
    }
    
    //创建文件夹
    std::string command;
    command = "mkdir -p " + output;
    system(command.c_str());
    
    //image_file.size()代表文件中总共的图片个数
    for (unsigned int frame = 0; frame < image_files.size(); ++frame) {
        cv::Mat image = cv::imread(image_files[frame]);
        std::cout << frame << std::endl;
        cv::imwrite(output+"/"+std::to_string(frame)+".jpg", image);
    }
    return 0;
}

/**
 * 图像操作
 */
int OpencvTest::imageOp(std::string path){
    cv::Mat src, gray_src;
    src = cv::imread(path);
    if(src.empty()){
        std::printf("could not load image ...");
        return -1;
    }
    
    cv::namedWindow("input", CV_WINDOW_AUTOSIZE);
    cv::imshow("input", src);
    
    /*
    //单通道修改像素值 反差效果
    cv::cvtColor(lib, gray_src, CV_BGR2GRAY);
    int rows = gray_src.rows;
    int cols = gray_src.cols;
    
    for (int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            int gray = gray_src.at<uchar>(row, col);
            gray_src.at<uchar>(row, col) = 255 - gray;  //反差效果
        }
    }
    
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", gray_src);
    */
    
    
    
    /*
    //三通道修改像素值 反差效果
    cv::Mat dst;
    dst.create(lib.size(), lib.type());
     
    int rows = lib.rows;
    int cols = lib.cols;
    int nc = lib.channels();
    
    for (int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            if (nc == 3){
                int b = lib.at<cv::Vec3b>(row,col)[0];
                int g = lib.at<cv::Vec3b>(row,col)[1];
                int r = lib.at<cv::Vec3b>(row,col)[2];

                dst.at<cv::Vec3b>(row,col)[0] = 255 - b;
                dst.at<cv::Vec3b>(row,col)[1] = 255 - g;
                dst.at<cv::Vec3b>(row,col)[2] = 255 - r;
                
            }
        }
    }
    
    //cv::bitwise_not(lib, dst);  //api方式
    
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", dst);
    */
    
    
    //三通道做灰度效果
    cv::cvtColor(src, gray_src, CV_BGR2GRAY);
    cv::Mat gray_src2 = gray_src.clone();
    cv::Mat dst;
    dst.create(src.size(), src.type());
    
    int rows = src.rows;
    int cols = src.cols;
    int nc = src.channels();
    
    for (int row = 0; row < rows; row++){
        for(int col = 0; col < cols; col++){
            if (nc == 3){
                int b = src.at<cv::Vec3b>(row,col)[0];
                int g = src.at<cv::Vec3b>(row,col)[1];
                int r = src.at<cv::Vec3b>(row,col)[2];

                //gray_src2.at<uchar>(row, col) = std::max(r, std::max(b,g));
                gray_src2.at<uchar>(row, col) = std::min(r, std::min(b,g));
            }
        }
    }
    
    
    cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    cv::imshow("output", gray_src);
    
    cv::namedWindow("output2", CV_WINDOW_AUTOSIZE);
    cv::imshow("output2", gray_src2);
    
    
    
    cv::waitKey();
    return 0;
}


/**
 * 图像混合
 */
int OpencvTest::imageMixture(std::string path1,std::string path2){
    cv::Mat src1,src2,dst;
    src1 = cv::imread(path1);
    src2 = cv::imread(path2);
    cv::imshow("src1", src1);
    cv::imshow("src2", src2);
    if(src1.empty() or src2.empty()){
        std::printf("could not load image ...");
        return -1;
    }
    
    if(src1.rows == src2.rows && src1.cols == src2.cols){
        double alpha = 0.5;
        //cv::addWeighted(src1, alpha, src2, 1-alpha, 0.0, dst);
        //cv::add(src1, src2, dst, cv::Mat());
        //cv::multiply(src1, src1, dst);
        cv::namedWindow("blend demo", CV_WINDOW_AUTOSIZE);
        cv::imshow("blend demo", dst);
    }else{
        std::printf("could not blend images ...");
        return -1;
    }
    

    cv::waitKey();
    return 0;
}


/**
 * 调整图像亮度和对比度
 */
int OpencvTest::adjustImageLuminanceL(std::string path){
    
    cv::Mat src,dst;
    src = cv::imread(path);
    if(src.empty()){
        std::printf("could not load image ...");
        return -1;
    }
    
    char input_win[] = "input image";
    //cv::cvtColor(lib, lib, CV_BGR2GRAY);
    cv::imshow(input_win, src);
    
    dst = cv::Mat::zeros(src.size(), src.type());
    
    float alpha = 1.1; //对比度
    float beta = 20;  //亮度
    
    //将CV_8UC3转换为CV_32F，然后获取像素点用cv::Vec3f  8位转32位精度更高
    src.convertTo(src, CV_32F);
    
    for(int row=0; row < src.rows; row++){
        for(int col = 0; col < src.cols; col++){
            if(src.channels() ==3){
                
                float b = src.at<cv::Vec3f>(row, col)[0];
                float g = src.at<cv::Vec3f>(row, col)[1];
                float r = src.at<cv::Vec3f>(row, col)[2];
                
                dst.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b*alpha + beta);
                dst.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g*alpha + beta);
                dst.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r*alpha + beta);
                
            }else if(src.channels() ==1){
                float v = src.at<uchar>(row, col);
                dst.at<uchar>(row, col) = cv::saturate_cast<uchar>(v*alpha + beta);
            }
        }
    }
    
    
    char output_win[] = "output image";
    cv::imshow(output_win, dst);
    cv::waitKey();
    return 0;
}


/**
 * 绘制图片和文字
 */
void myLines(cv::Mat bgImage){
    cv::Point p1 = cv::Point(20, 30);
    cv::Point p2;
    p2.x = 300;
    p2.y = 300;
    cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::line(bgImage, p1, p2, color, 1, cv::LINE_AA);   //LINE_AA 反锯齿 （渲染多了一步）
}

void myRectangle(cv::Mat bgImage){
    cv::Rect rect = cv::Rect(200, 100, 30, 30);
    cv::Scalar color = cv::Scalar(255, 0, 0);
    cv::rectangle(bgImage, rect, color, 5, cv::LINE_8);
}

void myEllipse(cv::Mat bgImage){
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::ellipse(bgImage, cv::Point(bgImage.cols/2, bgImage.rows/2),
                cv::Size(bgImage.cols/4, bgImage.rows/8),
                45, 0, 180, color, 2, cv::LINE_8);
}

void myCircle(cv::Mat bgImage){
    cv::Scalar color = cv::Scalar(0, 255, 255);
    cv::circle(bgImage, cv::Point(bgImage.cols/2, bgImage.rows/2),
               100, color, 2, cv::LINE_8);
}

void myPolygon(cv::Mat bgImage){
    cv::Point pts[1][5];
    pts[0][0] = cv::Point(100, 100);
    pts[0][1] = cv::Point(100, 200);
    pts[0][2] = cv::Point(200, 200);
    pts[0][3] = cv::Point(200, 100);
    pts[0][4] = cv::Point(100, 100);
    
    const cv::Point* ppts[] = {pts[0]};
    int npt[] = {5};
    cv::Scalar color = cv::Scalar(255, 12, 255);
    cv::fillPoly(bgImage, ppts, npt, 1, color, cv::LINE_8);
}

void randomLineDemo(cv::Mat bgImage){
    cv::RNG rng(12345);
    cv::Point pt1, pt2;
    cv::Mat bg = cv::Mat::zeros(bgImage.size(), bgImage.type());
    for(int i=0;i<100000;i++){
        pt1.x = rng.uniform(0, bg.cols); //正太分布随机数a
        pt2.x = rng.uniform(0, bg.cols);
        pt1.y = rng.uniform(0, bg.rows);
        pt2.y = rng.uniform(0, bg.rows);
        
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),rng.uniform(0, 255));
        if(cv::waitKey(50) > 0){
            break;
        }
        cv::line(bg, pt1, pt2, color, 1, 8);
        cv::imshow("random line demo", bg);
        
    }
    
}
    


int OpencvTest::drawText(std::string path){
    cv::Mat src,dst;
    src = cv::imread(path);
    if(src.empty()){
        std::printf("could not load images...");
        return -1;
    }
    
//    myLines(lib);
//    myRectangle(lib);
//    myEllipse(lib);
//    myCircle(lib);
//    myPolygon(lib);
//
//    cv::putText(lib, "Hello Opencv", cv::Point(100,300), CV_FONT_BLACK,
//                1.0, cv::Scalar(12, 255, 200), 2, 8);
//
//    const char input_win[] = "input";
//    cv::imshow(input_win, lib);
//    cv::waitKey();
  

    randomLineDemo(src);
    return 0;
}


/**
 * 模糊操作
 */
int OpencvTest::blurImage(std::string path){
    cv::Mat src,dst,dst2,dst3;
    src = cv::imread(path);
    if(src.empty()){
        std::printf("could not load images...");
        return -1;
    }

    const char input_win[] = "input";
    cv::imshow(input_win, src);
    
    
//    cv::blur(src, dst, cv::Size(11,11), cv::Point(-1, -1));
//    const char output_win[] = "output";
//    cv::imshow(output_win, dst);
//
//


    //中值模糊 3-核大小必须为奇数 方便取中指
    //作用: 1.椒盐噪声去噪点(去除斑点) 2.中值滤波后皮肤更光滑
    cv::medianBlur(src, dst, 3);
    const char medianBlur[] = "medianBlur";
    cv::imshow(medianBlur, dst);


    //双边模糊 15-卷积核 150-阈值(只处理大于阈值像素，可以提取更多的特征) 3-如果卷积核设为-1,那么卷积核的大小是根据这个值来设置的
    //1.磨皮效果（对比高斯模糊双边模糊保留了边缘信息）
    cv::bilateralFilter(src,dst2,15,100,5);
    const char bilateralFilter[] = "bilateralFilter";
    cv::imshow(bilateralFilter, dst);

    //掩膜操作-提升对比度
    cv::Mat resultImg;
    cv::Mat kernel = (cv::Mat_<int>(3,3) << 0, -1, 0 ,-1, 5, -1, 0, -1, 0);
    cv::filter2D(dst,resultImg, -1, kernel, cv::Point(-1, -1), 0);
    const char filter2D[] = "filter2D";
    cv::imshow(filter2D, resultImg);


    //高斯模糊
    cv::GaussianBlur(src, dst3, cv::Size(15,15), 5, 5);
    const char GaussianBlurt_win[] = "GaussianBlur";
    cv::imshow(GaussianBlurt_win, dst3);

    
   
    cv::waitKey();
    return 0;
}



/**
 * 结构元素调整
 */
int element_size = 3;
int max_size = 21;
const char OUTPUT_WIN[] = "output";
cv::Mat src, dst;

void CallBack_Demo(int, void*){
    int s = element_size * 2 + 1;
    cv::Mat structureElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(s, s), cv::Point(-1,-1));
    //膨胀
    //cv::dilate(src, dst,structureElement, cv::Point(-1,-1), 1);
    //腐蚀
    cv::erode(src, dst,structureElement);
    cv::imshow(OUTPUT_WIN, dst);
}

/**
 * 腐蚀和膨胀
 * 形态学基本操作：腐蚀、膨胀、开、闭
 * 膨胀：取Ksize核下的最大值，替换锚点覆盖下像素
 * 腐蚀：取Ksize核下的最小值，替换锚点覆盖下像素
 *
 * 腐蚀的用途：图像二值化后，需要提取大块的特征，用腐蚀手段将小的点去除
 * cv::Point(-1, -1)就是中心像素
 * @param path
 * @return
 */
int OpencvTest::corrosionAndSwell(std::string path) {

    src = cv::imread(path);
    cv::cvtColor(src, src, CV_BGR2GRAY);

    if(src.empty()) {
        std::printf("could not load image....");
        return -1;
    }

    const char input[] = "input";
    cv::imshow(input, src);

    cv::namedWindow(OUTPUT_WIN, CV_WINDOW_AUTOSIZE);

    cv::createTrackbar("Element Size:", OUTPUT_WIN, &element_size, max_size, CallBack_Demo);

    cv::waitKey();
    return 0;
}

