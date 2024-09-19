#include <iostream>
#include <chrono>
#include "manager.hpp"
int main(){

    std::string path_to_data = "/home/hakito/python_scripts/AirSim/Data3";
    std::shared_ptr<Manager> manager = std::make_shared<Manager>(path_to_data);
    
    for(int idx = 10; idx<manager->_data.flowList.itemCount()-10; ++idx){
        manager->computeEstimatedDepthMap(idx);
        manager->computeEstimatedFlow(idx);
        
        cv::Mat flow_debug_img, debug_depth_image, flow_residual_vis;
        manager->visualizeEstimatedDepthWithActualDepth(idx, debug_depth_image);
        manager->visualizeEstimatedFlow(idx, flow_debug_img);
        manager->visualizeFlowResidual(flow_residual_vis);
        

        cv::imshow("debug_depth_image", debug_depth_image);
        cv::imshow("flow_debug_img", flow_debug_img);
        cv::imshow("flow_residual_vis", flow_residual_vis);
        cv::waitKey(1);


        manager->propagateDepth();
    }
    return 0;
}