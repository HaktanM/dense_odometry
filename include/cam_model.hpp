#ifndef CAM_MODEL_HEADER
#define CAM_MODEL_HEADER

// Include standard libraries for std::cout manipulation
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip> // For std::fixed and std::setprecision

// include Ceres optimization library
#include "ceres/ceres.h"

// include Eigen
#include <Eigen/Eigen>
#include <Eigen/Core>

// include OpenCV for Visualization
#include <opencv2/opencv.hpp>

#include "utils.hpp"


// Camera Intrinsic and Extrinsic Calibration Parameters are stored.
class CamModel{
public:
    CamModel();
    // Intrinsic Parameters
    const int _width  = 640;
    const int _height = 512;
    double fx{320.0}, fy{320.0}, cx{320.0}, cy{256.0};
    Eigen::Matrix3d K, K_inv;

    // Extrinsic Parameters
    Eigen::Matrix3d R_cam_gimbal, R_gimbal_body, R_c_b, R_b_c;
    Eigen::Vector3d t_c_b, t_b_c;
    Eigen::Matrix4d T_c_b = Eigen::MatrixXd::Identity(4,4);
    Eigen::Matrix4d T_b_c = Eigen::MatrixXd::Identity(4,4);

    // Get Estimated Optical Flow From Estimated State and IMU measurements
    Eigen::MatrixXd getEstimatedOF(const Eigen::RowVectorXd vectorized_depth_map, const Eigen::Matrix3d R_c0_c1, const Eigen::Vector3d t_c0_c1) const;

    // Usefull Parameters
    Eigen::MatrixXd _vectorized_pixels   = Eigen::MatrixXd::Ones(3, _width*_height);
    Eigen::MatrixXd _vectorized_pixels2  = Eigen::MatrixXd::Zero(2, _width*_height);
    Eigen::MatrixXd _vectorized_bearings = Eigen::MatrixXd::Zero(3, _width*_height);

    void getVectorizedPixels();
}; // cam_model


#endif // CAM_MODEL_HEADER