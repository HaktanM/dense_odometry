#include "cam_model.hpp"

CamModel::CamModel(){

    // Initialize Intrinsic Parameters
    K << fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0;
    K_inv = K.inverse();

    // std::cout << K_inv << std::endl;

    // Initialize Extrinsic Parameters
    R_cam_gimbal << 0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0;

    R_gimbal_body << 0.0, 0.0, -1.0,
                        0.0, 1.0, 0.0,
                        1.0, 0.0, 0.0;

    // Camera Frame Expressed in Body Frame
    R_c_b = R_gimbal_body * R_cam_gimbal;       // The operator (*) is Matrix Multiplication in Eigen Library
    t_c_b << 0.0, 0.5, 0.0;
    T_c_b.block(0,0,3,3) = R_c_b;
    T_c_b.block(0,3,3,1) = t_c_b;

    
    // Body Frame Expressed in Camera Frame
    R_b_c = R_c_b.transpose();
    t_b_c = - R_b_c * t_c_b;                    // The operator (*) is Matrix Multiplication in Eigen Library
    T_b_c.block(0,0,3,3) = R_b_c;
    T_b_c.block(0,3,3,1) = t_b_c;

    // Compute Useful Parameters to Be Used Later
    getVectorizedPixels();
    _vectorized_bearings = K_inv * _vectorized_pixels;  // The operator (*) is Matrix Multiplication in Eigen Library
    _vectorized_pixels2  = _vectorized_pixels.topRows(2);
} // CamModel::getVectorizedPixels()


void CamModel::getVectorizedPixels(){
    for(int row=0; row<_height; ++row){
        for(int col=0; col<_width; ++col){
            _vectorized_pixels(0, row*_width + col) = (double)col + 0.5;  // The Pixel Coordinate is assigned to be middle of the pixel
            _vectorized_pixels(1, row*_width + col) = (double)row + 0.5;  // Hence we add 0.5
        }
    }

} // CamModel::getPixelMat()


Eigen::MatrixXd CamModel::getEstimatedOF(const Eigen::RowVectorXd vectorized_depth_map, const Eigen::Matrix3d R_c0_c1, const Eigen::Vector3d t_c0_c1) const{   

    ////////// Compute The Estimated Optical Flow //////////
    // get omega
    Eigen::MatrixXd KR = K * R_c0_c1;
    Eigen::MatrixXd    w_XY   =   KR.topRows(2) * _vectorized_bearings;
    Eigen::RowVectorXd w_Z    =     KR.row(2)   * _vectorized_bearings;;

    // get b
    Eigen::VectorXd b = K * t_c0_c1;
    Eigen::VectorXd b_XY = b.topRows(2);
    
    // Get Propagated Pixels
    Eigen::MatrixXd _propagated_pixels = ((w_XY.array().rowwise() * vectorized_depth_map.array()).colwise() + b_XY.array()).array().rowwise() / ((w_Z.array() * vectorized_depth_map.array()) + b(2,0)).array();

    // Get Estimated Flow
    Eigen::MatrixXd vectorized_estimated_flow = _propagated_pixels - _vectorized_pixels2;

    return vectorized_estimated_flow;
} // Eigen::MatrixXd CamModel::getEstimatedOF