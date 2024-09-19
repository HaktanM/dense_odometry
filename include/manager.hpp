#ifndef MANAGER_HEADER
#define MANAGER_HEADER

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>


#include "utils.hpp"
#include "cam_model.hpp"
#include "gpu_manager.hpp"


struct Manager
{
public:
    Manager(std::string path_to_data) : 
    _data(path_to_data), _width(_cam_model._width), _height(_cam_model._height), _totalPixCount(_width * _height),
    _bearings(_cam_model._vectorized_bearings.topRows(2).cast<float>()), _bearingPtr(_bearings.data()), 
    _pixelCoordinages(_cam_model._vectorized_pixels.topRows(2).cast<float>()), _pixelCoordPtr(_pixelCoordinages.data()),
    _GPUhandler(_pixelCoordPtr, _bearingPtr, _height, _width) // Construct _GPUhandler
    {
        _depth_byte_size = _totalPixCount * sizeof(float);
        _flow_byte_size  = 2 * _totalPixCount * sizeof(float);     // For each pixel, we have fx, fy. That's why we multiply by 2

        _depthPtr         = (float*)malloc(_depth_byte_size);      // Memory in Host
        _depthVariancePtr = (float*)malloc(_depth_byte_size);      // Memory in Host

        _estimatedFlowPtr = (float*)malloc(_flow_byte_size);       // Memory in Host
        _flowResidualPtr     = (float*)malloc(_flow_byte_size);       // Memory in Host 
        _flowJacobeanPtr     = (float*)malloc(_flow_byte_size * 6);
    }

    void getExtendedBodyPoseGT(int idx, Eigen::Matrix3d &R_b_g, Eigen::Vector3d &v_gb_g, Eigen::Vector3d &t_gb_g);
    void getIterativeCamPoseGT(int idx, Eigen::Matrix3d &R_c0_c1, Eigen::Vector3d &t_c0_c1);
    void computeEstimatedDepthMap(int idx);
    void computeEstimatedFlow(int idx);
    void computeFlowResidual(Eigen::Matrix3d R_c0_c1, Eigen::Vector3d t_c0_c1) const;
    void propagateDepth();

    // Visualization tools
    void visualizeEstimatedDepthWithActualDepth(int idx, cv::Mat &out);
    void visualizeEstimatedDepthWithActualDepth2(int idx, cv::Mat &out);
    void visualizeEstimatedFlow(int idx, cv::Mat &out);
    void visualizeFlowResidual(cv::Mat &out);

    /////////// WARNING ///////////
    // Do not change the order of the declared variables below. 
    
    // When initializing the Manager, we have
    // _data(path_to_data), _width(_cam_model._width), _height(_cam_model._height), _totalPixCount(_width * _height)          // Construct _data
    // _bearings(_cam_model._vectorized_bearings.topRows(2).cast<float>()), 
    // _bearingPtr(_bearings.data()), 
    // _pixelCoordinages(_cam_model._vectorized_pixels.topRows(2).cast<float>()), 
    // _pixelCoordPtr(_pixelCoordinages.data()),
    // _GPUhandler(_pixelCoordPtr, _bearingPtr, _height, _width) // Construct _GPUhandler

    // The order of initialization in the initializer list is based on 
    // the order of member declaration in the class, 
    // not the order in which they appear in the initializer list.

    ///////////////////////////// IMPORTANT /////////////////////////////
    /////////// DON'T MODIFY BEFORE READING THE WARNING BELOW ///////////

    CamModel _cam_model;
    DMU::dataHandler _data;
    int _width, _height, _totalPixCount; 
    Eigen::MatrixXf _bearings, _pixelCoordinages;

    float *_depthPtr;               // Pointer to store estimated depth map
    float *_depthVariancePtr;       // Pointer to store the variance of estimated depth map
    float *_bearingPtr;             // Bearing refers to (K_inv * pix). This is computed for every pixel once at the beginning of application. 
    float *_pixelCoordPtr;          // Pixel coordinages in image plane

    float *_estimatedFlowPtr;       // Pointer to hold estimated flows
    float *_flowResidualPtr;       // Pointer to hold flow residual
    float *_flowJacobeanPtr; 

    ManagerGPU _GPUhandler;
    int _depth_byte_size, _flow_byte_size;

    // Visualization parameters 
    double _depth_min{0.0}, _depth_max{50};
};


#endif // MANAGER_HEADER