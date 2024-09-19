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

#include "manager.hpp"

namespace SOLVER{

    // Flow constraint to get optimal iterative pose
    struct FlowConstraint{
    public:
    FlowConstraint(std::shared_ptr<Manager> manager, Eigen::Matrix3d R_c0_c1, Eigen::Vector3d t_c0_c1): 
    _manager(manager), _R_c0_c1(R_c0_c1), _t_c0_c1(t_c0_c1)
    {
        _residual_size = 2 * _manager->_cam_model._width * _manager->_cam_model._height;
    };

    // Operator to compute the Mahalanobis distance
    bool operator()(const double* const x, double* residual) const{
        Eigen::Vector3d psi(x[0], x[1], x[2]);
        Eigen::Matrix3d R_innovation = LU::exp_SO3(psi);
        Eigen::Vector3d t_innovation(x[3], x[4], x[5]);

        Eigen::Matrix3d R_c0_c1_innovated = _R_c0_c1 * R_innovation;
        Eigen::Vector3d t_c0_c1_innovated = _t_c0_c1 + t_innovation;

        _manager->computeFlowResidual(R_c0_c1_innovated, t_c0_c1_innovated);

        for (int i = 0; i < _residual_size; ++i) {
            residual[i] = static_cast<double>(_manager->_flowResidualPtr[i]);
        }
        return true;
    }

    private:    
    std::shared_ptr<Manager> _manager;
    Eigen::Matrix3d _K;
    Eigen::Matrix3d _R_c0_c1;
    Eigen::Vector3d _t_c0_c1;
    int _residual_size;
    }; // struct FlowConstraint  


} // namespace Optimizer