#include <iostream>
#include <chrono>
#include "manager.hpp"
int main(){

    std::string path_to_data = "/home/hakito/python_scripts/AirSim/Data3";
    std::shared_ptr<Manager> manager = std::make_shared<Manager>(path_to_data);
    
    int width = manager->_cam_model._width;
    int height = manager->_cam_model._height;
    int totalPixelCount = width * height;
    int flow_byte_size  = 2 * totalPixelCount * sizeof(float);

    // First feed information to the our manager
    for(int idx = 250; idx<320; ++idx){
        manager->computeEstimatedDepthMap(idx);
        manager->propagateDepth();
    }
    int idx = 119;
        

    // Get iterative cam pose
    Eigen::Matrix3d R_c0_c1;
    Eigen::Vector3d t_c0_c1;
    manager->getIterativeCamPoseGT(idx, R_c0_c1, t_c0_c1);
    manager->computeFlowResidual(R_c0_c1, t_c0_c1);

    // Seed the random number generator with the current time
    srand(static_cast<unsigned int>(time(0)));

    // Define the dimensions of the grid
    const int rows = 512;
    const int cols = 640;

    // Generate random row and column indices
    int random_row = rand() % rows;  // Random row between 0 and 511
    int random_col = rand() % cols;  // Random column between 0 and 639

    

    float** J = (float**)malloc(2 * totalPixelCount * sizeof(float*));
    
    for (int res_idx = 0; res_idx < 2*totalPixelCount; ++res_idx) {
        J[res_idx]     = (float*)malloc(6 * sizeof(float));
        J[res_idx + 1] = (float*)malloc(6 * sizeof(float));
        for (int state_idx = 0; state_idx < 6; ++state_idx) {
            J[res_idx][state_idx] = manager->_flowJacobeanPtr[res_idx*6 + state_idx];
        }
    }

    

    

    for(int trial_idx = 0; trial_idx<200; trial_idx++){
        int pixel_idx = random_row * width + random_col;
        float fx_dp_arr[6];
        float fy_dp_arr[6];

        for(int pert_idx = 0; pert_idx < 6; pert_idx++){
            double pert_amount = 1.0;

            Eigen::Matrix3f KR = (manager->_cam_model.K * R_c0_c1).cast<float>();
            Eigen::Vector3f b  = (manager->_cam_model.K * t_c0_c1).cast<float>();


            Eigen::Vector3d rot_pert(0.0, 0.0, 0.0);
            Eigen::Vector3d t_pert(0.0, 0.0, 0.0);

            if(pert_idx<3){
                pert_amount = 0.001;
                rot_pert[pert_idx] = pert_amount;
            }else{
                pert_amount = 0.01;
                t_pert[pert_idx-3] = pert_amount;
            }

            Eigen::Matrix3d R_perturbation = LU::exp_SO3(rot_pert);
            Eigen::Matrix3d R_perturbated = R_c0_c1 * R_perturbation;

            Eigen::Vector3d t_purturbated = t_c0_c1 + t_pert;

            float* KR_Ptr = KR.data();
            float* bPtr = b.data();

            Eigen::Matrix3f KR_p = (manager->_cam_model.K * R_perturbated).cast<float>();
            Eigen::Vector3f b_p  = (manager->_cam_model.K * t_purturbated).cast<float>();
            float* KR_Ptr_p      = KR_p.data();
            float* bPtr_p        = b_p.data();
            
            float *estimatedFlowPtr = (float*)malloc(flow_byte_size);
            float *purturbedFlowPtr = (float*)malloc(flow_byte_size);

            manager->_GPUhandler.getEstimatedFlow(estimatedFlowPtr, KR_Ptr, bPtr);
            manager->_GPUhandler.getEstimatedFlow(purturbedFlowPtr, KR_Ptr_p, bPtr_p);
            
            float fx_dp = (purturbedFlowPtr[pixel_idx*2] - estimatedFlowPtr[pixel_idx*2]) / pert_amount;
            float fy_dp = (purturbedFlowPtr[pixel_idx*2 + 1] - estimatedFlowPtr[pixel_idx*2 + 1]) / pert_amount;

            fx_dp_arr[pert_idx] = fx_dp;
            fy_dp_arr[pert_idx] = fy_dp;

            free(estimatedFlowPtr);
            free(purturbedFlowPtr);
        }
        

        std::cout << "Row: " << random_row << ". Column: " << random_col << std::endl;
        std::cout << "dfx / dstate  -- numeric versus analytic derivation" << std::endl;
        for(int i = 0; i<6; i++){
            std::cout << fx_dp_arr[i] << ", ";
        }
        std::cout << std::endl;

        for(int i = 0; i<6; i++){
            std::cout << J[2*pixel_idx][i] << ", ";
        }
        std::cout << std::endl  << std::endl;

        std::cout << "dfx / dstate  -- numeric versus analytic derivation" << std::endl;
        for(int i = 0; i<6; i++){
            std::cout << fy_dp_arr[i] << ", ";
        }
        std::cout << std::endl;
        for(int i = 0; i<6; i++){
            std::cout << J[2*pixel_idx + 1][i] << ", ";
        }
        std::cout << std::endl;

        
    }

    for (int res_idx = 0; res_idx < 2 * totalPixelCount; ++res_idx) {
        free(J[res_idx]);
    }
    free(J);

    return 0;
}