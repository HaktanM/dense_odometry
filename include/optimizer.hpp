#pragma once
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


#include "ceres/ceres.h"

#include "utils.hpp"

#include "cam_model.hpp"

namespace SOLVER{

    // Define Pixel Flow Constraint
    struct PixelFlow{
    public:
    PixelFlow(Eigen::Vector2d p, Eigen::Vector2d f, Eigen::Vector3d w, Eigen::Vector3d b) : _p(p), _f(f), _w(w), _b(b)
    {};

    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const d, double* residual) const {
        Eigen::Vector3d propagated_pixel = ( _w.topRows(2) * d[0] + _b.topRows(2) ) / (_w(2) * d[0] + _b(2));
        Eigen::Vector2d err = propagated_pixel.topRows(2) - _p -_f; 
        residual[0] = err[0];
        residual[1] = err[1];
        return true;
    }

    private:    
    Eigen::Vector2d _p, _f;
    Eigen::Vector3d _w,_b;   
    }; // struct PixelFlow

    // Define Optical Flow Constraint
    struct OpticalFlowConstraint{
    public:
    OpticalFlowConstraint(  
        const Eigen::MatrixXd& vectorized_flow,
        const Eigen::RowVectorXd& vectorized_depth_map,
        const Eigen::MatrixXd& Delta_R,
        const Eigen::VectorXd& Delta_t,
        const Eigen::MatrixXd& R_g_b,
        const Eigen::VectorXd& v_gb_b,
        const double &dT
    )
    : _vectorized_flow(vectorized_flow), _vectorized_depth_map(vectorized_depth_map), _Delta_R(Delta_R), _Delta_t(Delta_t), _R_g_b(R_g_b), _v_gb_b(v_gb_b), _dT(dT) 
    {};

    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const x, double* residual) const {
        // Eigen::Vector3d innovation_vector(x[0], x[1], x[2]);
        // Eigen::Vector3d v_innovated = _v_gb_b + innovation_vector;
        
        // Eigen::MatrixXd vectorized_estimated_flow = _cam_model.getEstimatedOF(_vectorized_depth_map, _Delta_R, _Delta_t, _R_g_b, v_innovated, _dT);
        // Eigen::MatrixXd flow_res = vectorized_estimated_flow - _vectorized_flow;

        // // residual[0] = (flow_res.row(0).array() / _vectorized_depth_map.array()).array().abs().sum() ;
        // // residual[1] = (flow_res.row(1).array() / _vectorized_depth_map.array()).array().abs().sum() ;
        // for(int pix_idx = 0; pix_idx<_vectorized_depth_map.cols(); ++pix_idx){
        //     int res_idx = 2*pix_idx;
        //     residual[res_idx]     = flow_res(0,pix_idx);
        //     residual[res_idx + 1] = flow_res(1,pix_idx);   
        // }
        // residual[0] = flow_res.array().abs().sum();
        return true;
    }

    private:
        CamModel _cam_model;
        Eigen::MatrixXd _vectorized_flow;     // Observed Optical Flow
        Eigen::RowVectorXd _vectorized_depth_map;         // Estimated or measured depth map
        Eigen::MatrixXd _Delta_R;           // Preintegrated IMU measurement for rotation
        Eigen::VectorXd _Delta_t;           // Preintegrated IMU measurement for translation
        Eigen::MatrixXd _R_g_b;             // Initial Camera Orientation w.r.t. global frame
        Eigen::VectorXd _v_gb_b;            // Inverse Camera Velocity w.r.t. global frame
        double _dT;                         // Elapsed Time Between Two Frames           
    }; // struct OpticalFlowConstraint

    // Define Innovation Cost
    struct InnovationCost{
    public:
    InnovationCost(){};
    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const x, double* residual) const {
        // Innovation Constraint
        residual[0] = x[0] * 100;
        residual[1] = x[1] * 100;
        residual[2] = x[2] * 100;
        return true;
    }
    }; // struct InnovationCost

    // Define Optical Flow Constraint
    struct InnovationDecomposer{
    public:
    InnovationDecomposer(  
        const Eigen::Vector3d& innovation,
        const Eigen::MatrixXd& R_g_b,
        const double &dT
    )
    : _innovation(innovation), _R_g_b(R_g_b), _dT(dT) 
    {};

    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const x, double* residual) const {

        // Eigen::Vector3d rot_inn(x[0], x[1], 0.0);       // Innovation Terms
        // Eigen::Vector3d vel_inn(x[2], x[3], x[4]);       // Innovation Terms
        // Eigen::Matrix3d R_inn = LU::exp_SO3(rot_inn);
        // Eigen::Vector3d expected_inn = vel_inn + 0.5 * _dT * _R_g_b * (R_inn - Eigen::MatrixXd::Identity(3,3)) * _cam_model.getGravity();
        // Eigen::Vector3d inn_res = expected_inn - _innovation;
        
        // // Innovation Constraint
        // residual[0] = inn_res[0];
        // residual[1] = inn_res[1];
        // residual[2] = inn_res[2];

        // // Cost for rotation innovation
        // residual[3] = x[0] * _rot_weight;
        // residual[4] = x[1] * _rot_weight;

        // // Cost for velocity innovation
        // residual[5] = x[2] * _vel_weight;
        // residual[6] = x[3] * _vel_weight;
        // residual[7] = x[4] * _vel_weight;

        return true;
    }

    private:
        CamModel _cam_model;
        Eigen::Vector3d _innovation;
        Eigen::Matrix3d _R_g_b;                             // Initial Camera Orientation w.r.t. global frame
        double _dT;                         // Elapsed Time Between Two Frames           
        double _rot_weight{3.0};
        double _vel_weight{1.0};
    }; // struct OpticalFlowConstraint
} // namespace Optimizer