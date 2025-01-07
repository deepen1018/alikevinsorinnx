# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
import torch
from alike import ALike, configs

# 導入 OpenCV 相關定義
cdef extern from "opencv2/core.hpp" namespace "cv":
    cdef cppclass Point2f:
        Point2f()
        Point2f(float x, float y)
        float x, y

    cdef cppclass Mat:
        Mat()
        int rows, cols
        unsigned char* data
        size_t step
        
cdef class PyALikeDetector:
    cdef:
        object alike_model
        
    def __cinit__(self):
        self.alike_model = ALike(**configs['alike-t'])
        
    def detect(self, np.ndarray[np.uint8_t, ndim=3] image,
              vector[Point2f]& points,
              int maxCorners,
              double qualityLevel,
              double minDistance,
              Mat& mask):
        
        # 轉換圖像格式
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 調用 ALIKE 檢測
        with torch.no_grad():
            results = self.alike_model(img_tensor)
            
        # 獲取關鍵點
        keypoints = results['keypoints']
        
        # 轉換結果格式
        for i in range(min(len(keypoints), maxCorners)):
            kp = keypoints[i]
            points.push_back(Point2f(float(kp[0]), float(kp[1])))

# 定義 C++ 類的實現
cdef extern from "alike_wrapper.h":
    cdef cppclass AlikeDetector:
        AlikeDetector()
        void detect(const Mat& image, vector[Point2f]& points,
                   int maxCorners, double qualityLevel,
                   double minDistance, const Mat& mask)

# 實現 C++ 類
cdef class PyALikeWrapper:
    cdef unique_ptr[AlikeDetector] c_detector
    
    def __cinit__(self):
        self.c_detector = unique_ptr[AlikeDetector](new AlikeDetector())
