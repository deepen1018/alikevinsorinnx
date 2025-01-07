// #pragma once
// #include <memory>
// #include <opencv2/opencv.hpp>

// class AlikeDetector {
// public:
//     AlikeDetector();
//     ~AlikeDetector();

//     void detect(const cv::Mat& image,
//                 std::vector<cv::Point2f>& points,
//                 int maxCorners,
//                 double qualityLevel,
//                 double minDistance,
//                 const cv::Mat& mask = cv::Mat());

// private:
//     class Impl;
//     std::unique_ptr<Impl> pimpl;
// };
#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

class AlikeDetector {
public:
    AlikeDetector();
    ~AlikeDetector();

    void detect(const cv::Mat& image,
                std::vector<cv::Point2f>& points,
                int maxCorners,
                double qualityLevel,
                double minDistance,
                //const cv::Mat& mask = cv::Mat()//can run
                const cv::Mat& mask,
                double& detection_time); 

private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
    
    
    void checkImageValidity(const cv::Mat& image) const;
};