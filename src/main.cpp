#include "manager.hpp"

int main(){

    std::string path_to_data = "/home/hakito/python_scripts/AirSim/Data3";
    Manager manager(path_to_data);

    for(int idx = 500; idx<manager._data.flowList.itemCount()-100; ++idx){
        manager.computeEstimatedDepthMap(idx);

        cv::Mat debug_img;
        manager.visualizeEstimatedDepthWithActualDepth(idx, debug_img);

        cv::imshow("debug_img", debug_img);
        cv::waitKey(1);

    }
    return 0;
}