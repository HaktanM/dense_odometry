#include <iostream>
#include <chrono>
#include "optimizer.hpp"
#include "manager.hpp"
int main(){

    std::string path_to_data = "/home/hakito/python_scripts/AirSim/Data3";
    
    // To pass the manager into optimization algorithm, I defined it as a shaed pointer.
    // Otherwise, passing mannager into the optimization structure as input takes 10 msec
    std::shared_ptr<Manager> manager = std::make_shared<Manager>(path_to_data);
    
    ceres::Solver::Options _options;
    _options.linear_solver_type = ceres::DENSE_QR;
    _options.minimizer_progress_to_stdout = false;
    _options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    _options.max_num_iterations = 200;
    _options.num_threads = 8;
    _options.min_trust_region_radius = 1e-12;

    const int residual_size = 2 * 512 * 640;

    

    for(int idx = 100; idx<manager->_data.flowList.itemCount()-100; ++idx){
        std::cout << idx << std::endl;
        // Get iterative cam pose
        Eigen::Matrix3d R_c0_c1;
        Eigen::Vector3d t_c0_c1;
        manager->getIterativeCamPoseGT(idx, R_c0_c1, t_c0_c1);
        
        /////// ADD NOISE TO TERMS ///////
        Eigen::VectorXd rotation_pert = Eigen::VectorXd::Random(3) * M_PI / 180.0 * 3.0;
        Eigen::Matrix3d R_pert = LU::exp_SO3(rotation_pert);

        Eigen::VectorXd t_pert = Eigen::VectorXd::Random(3) * 1.0;
        R_c0_c1 = R_c0_c1 * R_pert;
        t_c0_c1 = t_c0_c1 + t_pert;


        manager->computeEstimatedDepthMap(idx);
        manager->computeEstimatedFlow(idx);

        auto start = std::chrono::steady_clock::now();
        double x[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        ceres::CostFunction* flow_constraint = new ceres::NumericDiffCostFunction<SOLVER::FlowConstraint, ceres::CENTRAL, residual_size, 6>(
            new SOLVER::FlowConstraint(manager, R_c0_c1, t_c0_c1));

        ceres::Problem problem;
        ceres::Solver::Summary summary;
        problem.AddResidualBlock(flow_constraint, nullptr, x);

        auto end = std::chrono::steady_clock::now();
        auto elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Elapsed time for constructing the problem: " << elapsed_milliseconds.count() << " ms\n";


        start = std::chrono::steady_clock::now();
        ceres::Solve(_options, &problem, &summary);
        end = std::chrono::steady_clock::now();
        elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Elapsed time for solving it: " << elapsed_milliseconds.count() << " ms\n";


        if (summary.termination_type == ceres::CONVERGENCE) {
            std::cout << "CONVERGENCE" << std::endl;    
        } else{
            std::cout << "NOT CONVERGED" << std::endl;  
        }

        std::cout << "GT ROT: " << -rotation_pert[0] << " " << -rotation_pert[1] << " " << -rotation_pert[2] << std::endl;
        std::cout << "ES ROT: " << x[0] << " " << x[1] << " " << x[2] << std::endl;

        std::cout << "GT TR: " << -t_pert[0] << " " << -t_pert[1] << " " << -t_pert[2] << std::endl;
        std::cout << "ES TR: " << x[3] << " " << x[4] << " " << x[5] << std::endl << std::endl;
        
        // cv::Mat flow_debug_img, debug_depth_image;
        // manager->visualizeEstimatedFlow(idx, flow_debug_img);
        
        // cv::imshow("flow_debug_img", flow_debug_img);
        // cv::waitKey(1);

        manager->propagateDepth();
        std::cout << "--------------------" << std::endl;

    }
    return 0;
}