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
    float* bPtr = b.data();

    _GPUhandler.compute_depth_with_sigma(_depthPtr, _depthVariancePtr, flowPtr, KR_Ptr, bPtr);
}

void Manager::visualizeEstimatedDepthWithActualDepth(int idx, cv::Mat &out){
    std::string path_to_img_curr, path_to_depth, curr_timestamp_string;
    curr_timestamp_string       =   _data.flowList.getItemName(idx);
    path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
    path_to_depth               =   _data.depthList.getItemPathFromName(curr_timestamp_string);

    // Image Frame
    cv::Mat frame = cv::imread(path_to_img_curr);

    // Actual Depth Map
    cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
    cv::flip(depth_map, depth_map, 0);
    cv::Mat depth_map_vis = depth_map / 100.0 * 255.0;
    depth_map_vis.convertTo(depth_map_vis, CV_8U);
    cv::cvtColor(depth_map_vis, depth_map_vis, cv::COLOR_GRAY2BGR);
    cv::putText(depth_map_vis, "Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(150, 180, 125), 2);  

    // Estimated Depth Map
    cv::Mat estimated_depth(_height, _width, CV_32F, _depthPtr);
    cv::Mat estimated_depth_vis = estimated_depth / 100.0 * 255.0;
    estimated_depth_vis.convertTo(estimated_depth_vis, CV_8U);
    cv::cvtColor(estimated_depth_vis, estimated_depth_vis, cv::COLOR_GRAY2BGR);
    cv::putText(estimated_depth_vis, "Estimated Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(150, 180, 125), 2);

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
}