#include "manager.hpp"


void Manager::getExtendedBodyPoseGT(int idx, Eigen::Matrix3d &R_b_g, Eigen::Vector3d &v_gb_g, Eigen::Vector3d &t_gb_g){
    std::string curr_timestamp_string, next_timestamp_string;
    curr_timestamp_string       =   _data.flowList.getItemName(idx);
    long int curr_timestamp = std::stol(curr_timestamp_string);
    Eigen::MatrixXd curr_gt = _data.gt_data.at(curr_timestamp);

    R_b_g = curr_gt.block(0,1,1,9).reshaped(3,3).transpose();
    v_gb_g = curr_gt.block(0,10,1,3).reshaped(3,1);
    t_gb_g = curr_gt.block(0,13,1,3).reshaped(3,1);
}

void Manager::getIterativeCamPoseGT(int idx, Eigen::Matrix3d &R_c0_c1, Eigen::Vector3d &t_c0_c1){
    // Current pose of the body
    Eigen::Matrix3d R_b0_g;
    Eigen::Vector3d v_gb0_g, t_gb0_g;
    getExtendedBodyPoseGT(idx, R_b0_g, v_gb0_g, t_gb0_g);
    // Current pose of the camera
    Eigen::Matrix3d R_c0_g  = R_b0_g * _cam_model.R_c_b;
    Eigen::Vector3d t_gc0_g = R_b0_g * _cam_model.t_c_b + t_gb0_g;

    // Next pose of the cam
    Eigen::Matrix3d R_b1_g;
    Eigen::Vector3d v_gb1_g, t_gb1_g;
    getExtendedBodyPoseGT(idx + 1, R_b1_g, v_gb1_g, t_gb1_g);
    // Next pose of the cam
    Eigen::Matrix3d R_c1_g  = R_b1_g * _cam_model.R_c_b;
    Eigen::Vector3d t_gc1_g = R_b1_g * _cam_model.t_c_b + t_gb1_g;

    /// Compute the iterative cam pose
    Eigen::Matrix3d R_g_c1  = R_c1_g.transpose();
    R_c0_c1 = R_g_c1 * R_c0_g;
    t_c0_c1 = R_g_c1 * (t_gc0_g - t_gc1_g);
}

void Manager::computeEstimatedDepthMap(int idx){
    // Load the data
    std::string path_to_flow;
    path_to_flow = _data.flowList.getItemPath(idx);
    cv::Mat observed_flow = DMU::load_flow(path_to_flow);
    // Data to pointer
    float* flowPtr = observed_flow.ptr<float>();

    // Get iterative cam pose
    Eigen::Matrix3d R_c0_c1;
    Eigen::Vector3d t_c0_c1;
    getIterativeCamPoseGT(idx, R_c0_c1, t_c0_c1);

    Eigen::Matrix3f KR = (_cam_model.K * R_c0_c1).cast<float>();
    Eigen::Vector3f b  = (_cam_model.K * t_c0_c1).cast<float>();

    float* KR_Ptr = KR.data();
    float* bPtr   = b.data();

    _GPUhandler.refineDepthMap(_depthPtr, _depthVariancePtr, flowPtr, KR_Ptr, bPtr);
}

void Manager::propagateDepth(){
    _GPUhandler.propagateDepth();
}


void Manager::computeEstimatedFlow(int idx){

    // Get iterative cam pose
    Eigen::Matrix3d R_c0_c1;
    Eigen::Vector3d t_c0_c1;
    getIterativeCamPoseGT(idx, R_c0_c1, t_c0_c1);

    Eigen::Matrix3f KR = (_cam_model.K * R_c0_c1).cast<float>();
    Eigen::Vector3f b  = (_cam_model.K * t_c0_c1).cast<float>();

    float* KR_Ptr = KR.data();
    float* bPtr = b.data();

    _GPUhandler.getEstimatedFlow(_estimatedFlowPtr, KR_Ptr, bPtr);
}


void Manager::computeFlowResidual(Eigen::Matrix3d R_c0_c1, Eigen::Vector3d t_c0_c1) const{

    Eigen::Matrix3f K  = _cam_model.K.cast<float>();
    Eigen::Matrix3f KR = (_cam_model.K * R_c0_c1).cast<float>();
    Eigen::Vector3f b  = (_cam_model.K * t_c0_c1).cast<float>();

    float* K_Ptr      = K.data();
    float* KR_Ptr = KR.data();
    float* bPtr   = b.data();

    _GPUhandler.getFowResidual(_flowResidualPtr, _flowJacobeanPtr, K_Ptr, KR_Ptr, bPtr);
}

// void Manager::computeFlowResidual(int idx){
//     // Get iterative cam pose
//     Eigen::Matrix3d R_c0_c1;
//     Eigen::Vector3d t_c0_c1;
//     getIterativeCamPoseGT(idx, R_c0_c1, t_c0_c1);

//     Eigen::Matrix3f KR = (_cam_model.K * R_c0_c1).cast<float>();
//     Eigen::Vector3f b  = (_cam_model.K * t_c0_c1).cast<float>();

//     float* KR_Ptr = KR.data();
//     float* bPtr = b.data();

//     _GPUhandler.getFowResidual(_flowResidualPtr, KR_Ptr, bPtr);
// }

void Manager::visualizeEstimatedDepthWithActualDepth(int idx, cv::Mat &out){
    std::string path_to_flow, path_to_img_curr, path_to_depth, curr_timestamp_string;
    path_to_flow                =   _data.flowList.getItemPath(idx);
    curr_timestamp_string       =   _data.flowList.getItemName(idx);
    path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
    path_to_depth               =   _data.depthList.getItemPathFromName(curr_timestamp_string);

    // Image Frame
    cv::Mat frame = cv::imread(path_to_img_curr);

    // Load Actual Flow and Visualize
    cv::Mat actual_flow = DMU::load_flow(path_to_flow);
    cv::Mat actual_flow_bgr = VU::flowToColor(actual_flow);
    cv::putText(actual_flow_bgr, "Observed Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(10, 10, 10), 2);

    // Variables to store the min and max values
    double actual_depth_min, actual_depth_max, estimated_depth_min, estimated_depth_max;

    // Get Actual Depth Map
    cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
    cv::flip(depth_map, depth_map, 0);
    cv::minMaxLoc(depth_map, &actual_depth_min, &actual_depth_max);

    // Get Estimated Depth Map
    cv::Mat estimated_depth(_height, _width, CV_32F, _depthPtr);
    cv::minMaxLoc(depth_map, &estimated_depth_min, &estimated_depth_max);

    // Get the limits for normalization
    double depth_min = std::min(actual_depth_min, estimated_depth_min);
    double depth_max = std::max(actual_depth_max, estimated_depth_max);

    _depth_min = 0.99 * _depth_min + 0.01 * depth_min;
    _depth_max = 0.99 * _depth_max + 0.01 * depth_max;

    // Normalize actual depth map for visualization
    cv::Mat depth_map_tmp = (depth_map - _depth_min) / _depth_max * 255.0;
    depth_map_tmp.convertTo(depth_map_tmp, CV_8U);
    cv::Mat depth_map_vis;
    cv::applyColorMap(depth_map_tmp, depth_map_vis, cv::COLORMAP_TURBO);
    cv::putText(depth_map_vis, "Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 200), 2);

    // Normalize estimated depth map for visualization
    cv::Mat estimated_depth_tmp = (estimated_depth - _depth_min) / _depth_max * 255.0;
    estimated_depth_tmp.convertTo(estimated_depth_tmp, CV_8U);
    cv::Mat estimated_depth_vis;
    cv::applyColorMap(estimated_depth_tmp, estimated_depth_vis, cv::COLORMAP_TURBO);
    cv::putText(estimated_depth_vis, "Estimated Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 200), 2);


    // Covariance of the Estimated depth
    cv::Mat depth_variance(_height, _width, CV_32F, _depthVariancePtr);
    cv::Mat depth_variance_vis = depth_variance * 2 * 220;
    depth_variance_vis.convertTo(depth_variance_vis, CV_8U);
    cv::cvtColor(depth_variance_vis, depth_variance_vis, cv::COLOR_GRAY2BGR);
    cv::putText(depth_variance_vis, "Depth Variance", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(150, 180, 125), 2);

    // Visualize Depth Error
    cv::Mat depth_error = cv::abs(depth_map - estimated_depth);
    cv::Mat depthErrorPercentage;
    cv::divide(depth_error, depth_map, depthErrorPercentage);
    depthErrorPercentage = depthErrorPercentage * 100;
    depthErrorPercentage.convertTo(depthErrorPercentage, CV_8UC1);
    cv::Mat depth_error_percentage_vis;
    cv::applyColorMap(depthErrorPercentage, depth_error_percentage_vis, cv::COLORMAP_TURBO); 
    cv::putText(depth_error_percentage_vis, "Percentage Error Color Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(200, 200, 200), 2);

    cv::Mat out_up, out_down;

    cv::hconcat(frame, actual_flow_bgr, out_up);
    cv::hconcat(out_up, depth_variance_vis, out_up);

    cv::hconcat(depth_map_vis, estimated_depth_vis, out_down);
    cv::hconcat(out_down, depth_error_percentage_vis, out_down);
    cv::vconcat(out_up, out_down, out);


}

void Manager::visualizeEstimatedDepthWithActualDepth2(int idx, cv::Mat &out){
    std::string path_to_img_curr, path_to_depth, curr_timestamp_string;
    curr_timestamp_string       =   _data.flowList.getItemName(idx);
    path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
    path_to_depth               =   _data.depthList.getItemPathFromName(curr_timestamp_string);

    // Image Frame
    cv::Mat frame = cv::imread(path_to_img_curr);

    // Variables to store the min and max values
    double actual_depth_min, actual_depth_max, estimated_depth_min, estimated_depth_max;

    // Get Actual Depth Map
    cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
    cv::flip(depth_map, depth_map, 0);
    cv::minMaxLoc(depth_map, &actual_depth_min, &actual_depth_max);

    // Get Estimated Depth Map
    cv::Mat estimated_depth(_height, _width, CV_32F, _depthPtr);
    cv::minMaxLoc(depth_map, &estimated_depth_min, &estimated_depth_max);

    // Get the limits for normalization
    double depth_min = std::min(actual_depth_min, estimated_depth_min);
    double depth_max = std::max(actual_depth_max, estimated_depth_max);

    _depth_min = 0.99 * _depth_min + 0.01 * depth_min;
    _depth_max = 0.99 * _depth_max + 0.01 * depth_max;

    // Normalize actual depth map for visualization
    cv::Mat depth_map_tmp = (depth_map - _depth_min) / _depth_max * 255.0;
    depth_map_tmp.convertTo(depth_map_tmp, CV_8U);
    cv::Mat depth_map_vis;
    cv::applyColorMap(depth_map_tmp, depth_map_vis, cv::COLORMAP_TURBO);
    cv::putText(depth_map_vis, "Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 200), 2);

    // Normalize estimated depth map for visualization
    cv::Mat estimated_depth_tmp = (estimated_depth - _depth_min) / _depth_max * 255.0;
    estimated_depth_tmp.convertTo(estimated_depth_tmp, CV_8U);
    cv::Mat estimated_depth_vis;
    cv::applyColorMap(estimated_depth_tmp, estimated_depth_vis, cv::COLORMAP_TURBO);
    cv::putText(estimated_depth_vis, "Estimated Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 200), 2);


    // Covariance of the Estimated depth
    cv::Mat depth_variance(_height, _width, CV_32F, _depthVariancePtr);
    cv::Mat depth_variance_vis = depth_variance * 2 * 220;
    depth_variance_vis.convertTo(depth_variance_vis, CV_8U);
    cv::cvtColor(depth_variance_vis, depth_variance_vis, cv::COLOR_GRAY2BGR);
    cv::putText(depth_variance_vis, "Depth Variance", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(150, 180, 125), 2);

    // Depth Error
    std::vector<cv::Mat> histograms = VU::depthErrorHistogram(depth_map, estimated_depth, _totalPixCount);
    cv::Mat depth_err_hist_img = histograms.at(0);
    cv::Mat depth_err_perc_hist_img = histograms.at(1);

    cv::Mat out_up, out_down;

    cv::hconcat(frame, depth_variance_vis, out_up);
    cv::hconcat(out_up, depth_err_hist_img, out_up);

    cv::hconcat(depth_map_vis, estimated_depth_vis, out_down);
    cv::hconcat(out_down, depth_err_perc_hist_img, out_down);

    cv::vconcat(out_up, out_down, out);

    // cv::Mat depth_error = cv::abs(depth_map - estimated_depth);
    // cv::Mat depthErrorPercentage;
    // cv::divide(depth_error, depth_map, depthErrorPercentage);

    // depthErrorPercentage = depthErrorPercentage * 100;
    // depthErrorPercentage.convertTo(depthErrorPercentage, CV_8UC1);
    

    // cv::applyColorMap(depthErrorPercentage, out, cv::COLORMAP_TURBO); 
}

void Manager::visualizeEstimatedFlow(int idx, cv::Mat &out){
    std::string path_to_flow, path_to_img_curr, path_to_img_next, path_to_depth, curr_timestamp_string, next_timestamp_string;
    path_to_flow                =   _data.flowList.getItemPath(idx);
    curr_timestamp_string       =   _data.flowList.getItemName(idx);
    next_timestamp_string       =   _data.flowList.getItemName(idx+1);
    path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
    path_to_img_next            =   _data.imgList.getItemPathFromName(next_timestamp_string);

    long int curr_timestamp = std::stol(curr_timestamp_string);
    long int next_timestamp = std::stol(next_timestamp_string);


    // Load Current Image Frame for visualizaiton
    cv::Mat frame = cv::imread(path_to_img_curr);

    // Load Actual Flow for Comparison Flow
    cv::Mat actual_flow = DMU::load_flow(path_to_flow);

    // Get the estimated flow
    cv::Mat estimated_flow(_height, _width, CV_32FC2, _estimatedFlowPtr);

    // Visualize the flows
    cv::Mat actual_flow_bgr = VU::flowToColor(actual_flow);
    cv::putText(actual_flow_bgr, "Observed Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(10, 10, 10), 2);

    cv::Mat estimated_flow_bgr = VU::flowToColor(estimated_flow);
    cv::putText(estimated_flow_bgr, "Estimated Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(10, 10, 10), 2);

    //// Warp frame to see warping error
    cv::Mat curr_frame = cv::imread(path_to_img_curr, cv::IMREAD_GRAYSCALE);
    cv::Mat next_frame = cv::imread(path_to_img_next, cv::IMREAD_GRAYSCALE);

    curr_frame.convertTo(curr_frame, CV_64F);
    next_frame.convertTo(next_frame, CV_64F);

    // Warp using observed flow
    cv::Mat warped_frame = VU::warpFlow(curr_frame, actual_flow);
    cv::Mat warping_error = next_frame - warped_frame;
    warping_error = cv::abs(warping_error);
    warping_error.convertTo(warping_error, CV_8U);
    cv::cvtColor(warping_error, warping_error, cv::COLOR_GRAY2BGR);
    cv::putText(warping_error, "Warping Error of Observed Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(50, 250, 250), 2);

    // Warp using estimated flow
    cv::Mat warped_frame_es = VU::warpFlow(curr_frame, estimated_flow);
    cv::Mat warping_error_es = next_frame - warped_frame_es;
    warping_error_es = cv::abs(warping_error_es);
    warping_error_es.convertTo(warping_error_es, CV_8U);
    cv::cvtColor(warping_error_es, warping_error_es, cv::COLOR_GRAY2BGR);
    cv::putText(warping_error_es, "Warping Error of Estimated Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(50, 250, 250), 2);

    // Concatenate the frame for better visualization
    cv::Mat out_up, out_down;

    cv::hconcat(actual_flow_bgr, estimated_flow_bgr, out_up);
    cv::hconcat(warping_error, warping_error_es, out_down);
    cv::vconcat(out_up, out_down, out);
}



void Manager::visualizeFlowResidual(cv::Mat &out){
    // Get the estimated flow
    cv::Mat flow_residual(_height, _width, CV_32FC2, _flowResidualPtr);

    cv::Mat flow_residual_bgr = VU::flowToColor(flow_residual);
    cv::putText(flow_residual_bgr, "Flow Residual", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(10, 10, 10), 2);

    out = flow_residual_bgr;
}