cmake_minimum_required(VERSION 3.10)
project(vins)

# CMake ????
cmake_policy(SET CMP0048 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ?? Python3 ? NumPy ??
set(Python3_EXECUTABLE /usr/bin/python3)
set(Python3_INCLUDE_DIR /usr/include/python3.8)
set(Python3_LIBRARY "/usr/lib/aarch64-linux-gnu/libpython3.8.so")
set(Python3_NumPy_INCLUDE_DIRS "/home/jetsonnx/.local/lib/python3.8/site-packages/numpy/core/include")

set(camera_models_DIR "/home/jetsonnx/deepen_ws/devel/share/camera_models/cmake")
list(APPEND CMAKE_PREFIX_PATH ${camera_models_DIR})

# ??????
find_package(Python3 REQUIRED COMPONENTS Development NumPy)
find_package(Cython REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport
)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# ?????????
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # ????

# ????
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

find_package(Eigen3)
catkin_package()

# ?? alike_wrapper ???
add_library(alike_wrapper SHARED
    src/alike_detector/alike_wrapper.cpp
)

target_include_directories(alike_wrapper PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alike_detector
)

target_link_libraries(alike_wrapper PUBLIC
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${Python3_LIBRARIES}
)

# VINS ??
add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
)

target_link_libraries(vins_lib PUBLIC
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    alike_wrapper
)

# ?????
add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib)

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
target_link_libraries(kitti_odom_test vins_lib)

add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
target_link_libraries(kitti_gps_test vins_lib)

# ????
install(TARGETS alike_wrapper
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alike_detector/alike_wrapper.py
    ${CMAKE_CURRENT_SOURCE_DIR}/src/alike_detector/__init__.py
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/src/alike_detector
)
