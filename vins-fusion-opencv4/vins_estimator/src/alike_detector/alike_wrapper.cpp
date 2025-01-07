#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "alike_wrapper.h"
#include <ros/ros.h>

class AlikeDetector::Impl {
public:
    static bool python_initialized;
    PyThreadState *_save;
    PyObject* alike_detector;

    Impl() : alike_detector(nullptr) {
        try {
            ROS_INFO("Initializing ALIKE detector implementation");
            if (!python_initialized) {
                ROS_INFO("Initializing Python interpreter");
                Py_Initialize();
                if (!Py_IsInitialized()) {
                    throw std::runtime_error("Failed to initialize Python interpreter");
                }
                PyEval_InitThreads();
                python_initialized = true;
            }
            ROS_INFO("Initializing NumPy arrays");
            _import_array();
            _save = PyEval_SaveThread();
            ROS_INFO("Implementation initialized successfully");
        }
        catch (const std::exception& e) {
            ROS_ERROR("Implementation initialization failed: %s", e.what());
            throw;
        }
    }

    ~Impl() {
        PyEval_RestoreThread(_save);
        if (alike_detector) {
            Py_DECREF(alike_detector);
            alike_detector = nullptr;
        }
        if (python_initialized) {
            Py_Finalize();
            python_initialized = false;
        }
    }

    bool init() {
        try {
            PyEval_RestoreThread(_save);
            ROS_INFO("Starting ALIKE detector initialization");

            PyRun_SimpleString(R"(
import sys
import os

sys.path.insert(0, '/home/jetsonnx/deepen_ws/src/vins-fusion-opencv4/vins_estimator/src/alike_detector')
sys.path.insert(0, '/home/jetsonnx/deepen_ws/src/vins-fusion-opencv4/vins_estimator/src')
print('Python path:', sys.path)
)");

            importRequiredModules();
            createDetectorInstance();
            _save = PyEval_SaveThread();
            ROS_INFO("ALIKE detector initialized successfully");
            return true;
        }
        catch (const std::exception& e) {
            _save = PyEval_SaveThread();
            ROS_ERROR("ALIKE initialization error: %s", e.what());
            return false;
        }
    }

private:
    void importRequiredModules() {
        ROS_INFO("Importing required Python modules");
        PyObject* numpy = PyImport_ImportModule("numpy");
        if (!numpy) {
            PyErr_Print();
            throw std::runtime_error("Cannot import numpy");
        }
        Py_DECREF(numpy);
        ROS_INFO("Successfully imported numpy");

        PyObject* module = PyImport_ImportModule("alike_wrapper");
        if (!module) {
            PyErr_Print();
            throw std::runtime_error("Cannot import alike_wrapper");
        }
        Py_DECREF(module);
        ROS_INFO("All required modules imported successfully");
    }

    void createDetectorInstance() {
        ROS_INFO("Creating ALIKE detector instance");
        PyObject* module = PyImport_ImportModule("alike_wrapper");
        if (!module) {
            throw std::runtime_error("Module not found when creating detector");
        }

        PyObject* detector_class = PyObject_GetAttrString(module, "PyALikeDetector");
        if (!detector_class) {
            Py_DECREF(module);
            throw std::runtime_error("Cannot find PyALikeDetector class");
        }

        alike_detector = PyObject_CallObject(detector_class, NULL);
        if (!alike_detector) {
            Py_DECREF(detector_class);
            Py_DECREF(module);
            PyErr_Print();
            throw std::runtime_error("Cannot create PyALikeDetector instance");
        }

        Py_DECREF(detector_class);
        Py_DECREF(module);
        ROS_INFO("Detector instance created successfully");
    }
};

bool AlikeDetector::Impl::python_initialized = false;

AlikeDetector::AlikeDetector() : pimpl(new Impl()) {
    ROS_INFO("Starting ALIKE detector construction");
    if (!pimpl->init()) {
        throw std::runtime_error("Failed to initialize ALIKE detector");
    }
    ROS_INFO("ALIKE detector constructed successfully");
}

AlikeDetector::~AlikeDetector() = default;

cv::Mat processImageForModel(const cv::Mat& input_image) {
    if (input_image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    cv::Mat processed_image;
    if (input_image.channels() == 1) {
        cv::cvtColor(input_image, processed_image, cv::COLOR_GRAY2RGB);
    } else if (input_image.channels() == 3) {
        processed_image = input_image.clone();
    } else {
        throw std::runtime_error("Unsupported image channels");
    }
    return processed_image;
}

PyObject* createNumPyArray(const cv::Mat& rgb_img) {
    if (!rgb_img.isContinuous()) {
        throw std::runtime_error("Input image is not continuous");
    }
    npy_intp dims[3] = {rgb_img.rows, rgb_img.cols, rgb_img.channels()};
    PyObject* array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, rgb_img.data);
    if (!array) {
        throw std::runtime_error("Failed to create NumPy array");
    }
    return array;
}

void AlikeDetector::detect(const cv::Mat& image,
                           std::vector<cv::Point2f>& points,
                           int maxCorners,
                           double qualityLevel,
                           double minDistance,
                           const cv::Mat& mask,
                           double& detection_time) {
    try {
        ROS_INFO("Starting ALIKE detection");

        // ??????
        if (image.empty()) {
            throw std::runtime_error("Input image is empty");
        }

        // ??????? RGB
        cv::Mat rgb_img = processImageForModel(image);
        ROS_INFO("Image processed: size=%dx%d, channels=%d", rgb_img.cols, rgb_img.rows, rgb_img.channels());

        // ???????????????
        cv::Mat resized_img;
        cv::resize(rgb_img, resized_img, cv::Size(rgb_img.cols / 2, rgb_img.rows / 2));
        ROS_INFO("Resized image: size=%dx%d", resized_img.cols, resized_img.rows);

        // ?? NumPy ??
        npy_intp dims[3] = {resized_img.rows, resized_img.cols, resized_img.channels()};
        PyObject* array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, resized_img.data);
        if (!array) {
            throw std::runtime_error("Failed to create NumPy array");
        }

        // ?? Python ????
        PyObject* result = PyObject_CallMethod(pimpl->alike_detector, "detect", "(Oidd)", array, maxCorners, qualityLevel, minDistance);
        Py_DECREF(array);

        if (!result || PyErr_Occurred()) {
            PyErr_Print();
            throw std::runtime_error("Python detection failed");
        }

        // ????? NumPy ??
        PyArrayObject* points_array = reinterpret_cast<PyArrayObject*>(PyTuple_GetItem(result, 0));
        detection_time = PyFloat_AsDouble(PyTuple_GetItem(result, 1));

        if (points_array && PyArray_Check(points_array)) {
            int n_points = std::min(maxCorners, (int)PyArray_DIM(points_array, 0));
            float* points_data = static_cast<float*>(PyArray_DATA(points_array));

            points.clear();
            for (int i = 0; i < n_points; i++) {
                float x = (points_data[i * 2] + 1.0f) * resized_img.cols / 2.0f;
                float y = (points_data[i * 2 + 1] + 1.0f) * resized_img.rows / 2.0f;
                points.emplace_back(cv::Point2f(x, y));
            }
        }

        Py_DECREF(result);
        ROS_INFO("Detection completed. Points detected: %zu", points.size());
    } catch (const std::exception& e) {
        ROS_ERROR("Detection error: %s", e.what());
        points.clear();
        throw;
    }
}

