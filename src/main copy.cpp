// #include <iostream>
// #include <opencv2/opencv.hpp>


// #include <Eigen/Dense>

// #include <filesystem> 
// #include <fstream>
// #include <iostream>
// #include <vector>

// #include <opencv2/opencv.hpp>

// #include "optimizer.hpp"
// #include "utils.hpp"
// #include "cam_model.hpp"
// #include <chrono>


// #include <opencv2/core/eigen.hpp>

// #define EIGEN_USE_THREADS
// #include <omp.h>



// cv::Mat get_OF(CamModel cam_model, cv::Mat depth_map, Eigen::MatrixXd const gt_data, Eigen::MatrixXd const imu_data, double dT){
//     // Depth map should have dtype CV_64F
//     depth_map.convertTo(depth_map, CV_64F);

//     // Convert the depth_map into an Eigen Vector
//     // cv::transpose(depth_map, depth_map);
//     Eigen::RowVectorXd vectorized_depth_map = Eigen::Map<Eigen::RowVectorXd>(depth_map.ptr<double>(), depth_map.rows*depth_map.cols);

//     // Get GT
//     Eigen::Matrix3d R_g_b = gt_data.block(0,1,1,9).reshaped(3,3);
//     Eigen::Vector3d v_gb_g = gt_data.block(0,10,1,3).reshaped(3,1);
//     Eigen::Vector3d v_gb_b = R_g_b * v_gb_g;

//     // Get preintegrated IMU measurement
//     Eigen::Matrix3d Delta_R = imu_data.block(0,1,1,9).reshaped(3,3).transpose();
//     Eigen::Vector3d Delta_t = imu_data.block(0,13,1,3).reshaped(3,1);

//     // std::cout << imu_data.block(0,1,1,9) <<std::endl;
//     // std::cout << Delta_R << std::endl;
//     // std::cout << R_b_g << std::endl;
//     // std::cout << v_gb_g << std::endl;
//     // std::cout << Delta_R << std::endl;
//     // std::cout << Delta_t << std::endl;

//     Eigen::MatrixXd vectorized_of = cam_model.getEstimatedOF(vectorized_depth_map, Delta_R, Delta_t, R_g_b, v_gb_b, dT);
//     Eigen::MatrixXd vectorized_flow_x = vectorized_of.row(0);
//     Eigen::MatrixXd vectorized_flow_y = vectorized_of.row(1);

//     // std::cout << vectorized_flow_y << std::endl;

//     // Convert Flow into OpenCV cv::Mat object
//     cv::Mat flow_x, flow_y;
//     flow_x = cv::Mat(cam_model._height, cam_model._width, CV_64F, vectorized_flow_x.data()).clone(); 
//     flow_y = cv::Mat(cam_model._height, cam_model._width, CV_64F, vectorized_flow_y.data()).clone(); 

//     // Create a vector to hold the channels
//     std::vector<cv::Mat> channels = { flow_x, flow_y };

//     // Merge channels into a single multi-channel Mat
//     cv::Mat flow;
//     cv::merge(channels, flow);

//     return flow;
// }

// int main() {   

//     // Initialize our cam model
//     CamModel cam_model;

//     // Get the data
//     std::string main_path = "/home/hakito/python_scripts/AirSim/Data3";
//     DMU::dataHandler _data(main_path);
//     for(int idx = 20; idx<_data.flowList.itemCount()-2; ++idx){ //

//         // Get the related data    
//         std::string path_to_flow, path_to_img_curr, path_to_img_next, path_to_depth, curr_timestamp_string, next_timestamp_string;
//         path_to_flow                =   _data.flowList.getItemPath(idx);
//         curr_timestamp_string       =   _data.flowList.getItemName(idx);
//         next_timestamp_string       =   _data.flowList.getItemName(idx+1);
//         path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
//         path_to_img_next            =   _data.imgList.getItemPathFromName(next_timestamp_string);
//         path_to_depth               =   _data.depthList.getItemPathFromName(curr_timestamp_string);


//         long int curr_timestamp = std::stol(curr_timestamp_string);
//         long int next_timestamp = std::stol(next_timestamp_string);
//         double dT = ((double)(next_timestamp - curr_timestamp))*(1e-9);  // Unit should be seconds

//         Eigen::MatrixXd preintegrated_imu = _data.imu_data.at(curr_timestamp);
//         Eigen::MatrixXd curr_gt = _data.gt_data.at(curr_timestamp);
        
//         // Load Flow
//         cv::Mat flow = DMU::load_flow(path_to_flow);

//         // Load Image
//         cv::Mat frame = cv::imread(path_to_img_curr);

//         // Load Depth Map
//         cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
//         cv::flip(depth_map, depth_map, 0);

//         // Get Estimated Optical Flow
//         cv::Mat gt_of = get_OF(cam_model, depth_map, curr_gt, preintegrated_imu, dT);

//         // break;

//         ///////////////////// VISUALIZATIONS /////////////////////
//         // Warp the current frame to next frame
//         cv::Mat curr_frame = cv::imread(path_to_img_curr, cv::IMREAD_GRAYSCALE);
//         cv::Mat next_frame = cv::imread(path_to_img_next, cv::IMREAD_GRAYSCALE);

//         curr_frame.convertTo(curr_frame, CV_64F);
//         next_frame.convertTo(next_frame, CV_64F);
//         cv::Mat warped_frame = VU::warpFlow(curr_frame, flow);
//         cv::Mat warping_error = next_frame - warped_frame;

//         cv::Mat warped_frame_es = VU::warpFlow(curr_frame, gt_of);
//         cv::Mat warping_error_es = next_frame - warped_frame_es;

//         ///////// VISUALIZE ///////////
//         cv::Mat flow_bgr = VU::flowToColor(flow);
//         cv::Mat estimated_flow_bgr = VU::flowToColor(gt_of);
//         cv::Mat depth_map_vis = depth_map / 100.0 * 255.0;
//         depth_map_vis.convertTo(depth_map_vis, CV_8U);
//         warped_frame.convertTo(warped_frame, CV_8U);

//         warping_error_es = cv::abs(warping_error_es);
//         warping_error_es.convertTo(warping_error_es, CV_8U);

//         cv::Mat warping_error_vis = cv::abs(warping_error);
//         warping_error_vis.convertTo(warping_error_vis, CV_8U);

    
//         // Display the color wheel image
//         cv::imshow("Observed Flow", flow_bgr);
//         cv::imshow("Estimated Flow", estimated_flow_bgr);
//         cv::imshow("Depth Map", depth_map_vis);
//         cv::imshow("warped_frame", warped_frame);
//         cv::imshow("warping_error_es", warping_error_es);
//         cv::imshow("warping_error_vis", warping_error_vis);
//         cv::imshow("Frame", frame);
//         cv::waitKey(1);
//     }
    
// }