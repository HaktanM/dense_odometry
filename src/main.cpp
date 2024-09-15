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



// cv::Mat EigenVectorFlow2cvMat(Eigen::MatrixXd vectorized_flow, int width, int height){
//     Eigen::MatrixXd vectorized_flow_x = vectorized_flow.row(0);
//     Eigen::MatrixXd vectorized_flow_y = vectorized_flow.row(1);

//     // Convert Flow into OpenCV cv::Mat object
//     cv::Mat flow_x, flow_y;
//     flow_x = cv::Mat(height, width, CV_64F, vectorized_flow_x.data()).clone(); 
//     flow_y = cv::Mat(height, width, CV_64F, vectorized_flow_y.data()).clone(); 

//     // Create a vector to hold the channels
//     std::vector<cv::Mat> channels = { flow_x, flow_y };

//     // Merge channels into a single multi-channel Mat
//     cv::Mat flow;
//     cv::merge(channels, flow);
//     return flow;
// }



// void visualize_data(DMU::dataHandler _data, int idx, Eigen::MatrixXd vectorized_estimated_flow){
//     // Get the related data    
//     std::string path_to_flow, path_to_img_curr, path_to_img_next, path_to_depth, curr_timestamp_string, next_timestamp_string;
//     path_to_flow                =   _data.flowList.getItemPath(idx);
//     curr_timestamp_string       =   _data.flowList.getItemName(idx);
//     next_timestamp_string       =   _data.flowList.getItemName(idx+1);
//     path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
//     path_to_img_next            =   _data.imgList.getItemPathFromName(next_timestamp_string);
//     path_to_depth               =   _data.depthList.getItemPathFromName(curr_timestamp_string);

//     // Load Image
//     cv::Mat frame = cv::imread(path_to_img_curr);

//     // Load the grayscale images
//     cv::Mat curr_frame = cv::imread(path_to_img_curr, cv::IMREAD_GRAYSCALE);
//     cv::Mat next_frame = cv::imread(path_to_img_next, cv::IMREAD_GRAYSCALE);

//     // Load Observed Flow
//     cv::Mat observed_flow = DMU::load_flow(path_to_flow);

//     // Load Depth Map
//     cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
//     cv::flip(depth_map, depth_map, 0);

//     // Get the estimated flow
//     cv::Mat estimated_flow = EigenVectorFlow2cvMat(vectorized_estimated_flow, curr_frame.cols, curr_frame.rows);

//     // Compute warping frames using estimated and observed flows
//     curr_frame.convertTo(curr_frame, CV_64F);
//     next_frame.convertTo(next_frame, CV_64F);
//     cv::Mat warped_frame_ob = VU::warpFlow(curr_frame, observed_flow);
//     cv::Mat warping_error_ob = next_frame - warped_frame_ob;

//     cv::Mat warped_frame_es = VU::warpFlow(curr_frame, estimated_flow);
//     cv::Mat warping_error_es = next_frame - warped_frame_es;

//     cv::Mat warping_error_es_vis = cv::abs(warping_error_es);
//     warping_error_es_vis.convertTo(warping_error_es_vis, CV_8U);
//     cv::cvtColor(warping_error_es_vis, warping_error_es_vis, cv::COLOR_GRAY2BGR);
//     cv::putText(warping_error_es_vis, "Estimated Warping Error", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 255), 2);

//     cv::Mat warping_error_ob_vis = cv::abs(warping_error_ob);
//     warping_error_ob_vis.convertTo(warping_error_ob_vis, CV_8U);
//     cv::cvtColor(warping_error_ob_vis, warping_error_ob_vis, cv::COLOR_GRAY2BGR);
//     cv::putText(warping_error_ob_vis, "Observed Warping Error", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 100, 255), 2);


//     // ///////// VISUALIZE ///////////
//     cv::Mat observed_flow_bgr = VU::flowToColor(observed_flow);
//     cv::putText(observed_flow_bgr, "Observed Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

//     cv::Mat estimated_flow_bgr = VU::flowToColor(estimated_flow);
//     cv::putText(estimated_flow_bgr, "Estimated Flow", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

//     cv::Mat depth_map_vis = depth_map / 100.0 * 255.0;
//     depth_map_vis.convertTo(depth_map_vis, CV_8U);
//     cv::cvtColor(depth_map_vis, depth_map_vis, cv::COLOR_GRAY2BGR);
//     cv::putText(depth_map_vis, "Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 255, 100), 2);



//     // Create the destination matrix
//     cv::Mat flow_comparison, warping_error_comparisons, inpur_vis, debug_img;
    
//     // Concatenate horizontally
//     cv::hconcat(estimated_flow_bgr, observed_flow_bgr, flow_comparison);
//     cv::hconcat(warping_error_es_vis, warping_error_ob_vis, warping_error_comparisons);
//     cv::hconcat(frame, depth_map_vis, inpur_vis);
//     cv::vconcat(flow_comparison, warping_error_comparisons, debug_img);


//     // // Display the color wheel image
//     cv::imshow("Debug Image", debug_img);

//     cv::imshow("Input Frames", inpur_vis);
//     // cv::imshow("Frame", frame);
//     cv::waitKey(1);
// }



// Eigen::MatrixXd cvFlow2Eigen(cv::Mat flow){
//      // Vector to hold the two channels
//     std::vector<cv::Mat> flow_channels(2);

//     // flow datatype should be double
//     flow.convertTo(flow, CV_64F);

//     // Split the flow into its channels
//     cv::split(flow, flow_channels);

//     // cv::transpose(flow_channels[0], flow_channels[0]);
//     // cv::transpose(flow_channels[1], flow_channels[1]);

//     // Vectorize observed flow
//     Eigen::MatrixXd vectorized_flow_x = Eigen::Map<Eigen::RowVectorXd>(flow_channels[0].ptr<double>(), flow_channels[0].rows*flow_channels[0].cols);
//     Eigen::MatrixXd vectorized_flow_y = Eigen::Map<Eigen::RowVectorXd>(flow_channels[1].ptr<double>(), flow_channels[1].rows*flow_channels[1].cols);

//     Eigen::MatrixXd vectorized_flow(2, vectorized_flow_x.cols());
//     vectorized_flow << vectorized_flow_x, vectorized_flow_y;
//     return vectorized_flow;
// }


// int main() {   

//     // Initialize our cam model
//     CamModel cam_model;

//     // Get the data
//     std::string main_path = "/home/hakito/python_scripts/AirSim/Data3";
//     DMU::dataHandler _data(main_path);
//     for(int idx = 500; idx<_data.flowList.itemCount()-2; ++idx){ //
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

//         Eigen::MatrixXd curr_gt = _data.gt_data.at(curr_timestamp);
//         Eigen::MatrixXd next_gt = _data.gt_data.at(next_timestamp);
        

//         /// Get the pose of the camera with respect to global navigation frame
//         // Current pose of the body and camera
//         Eigen::Matrix3d R_b0_g = curr_gt.block(0,1,1,9).reshaped(3,3).transpose();
//         Eigen::Vector3d v_gb0_g = curr_gt.block(0,10,1,3).reshaped(3,1);
//         Eigen::Vector3d t_gb0_g = curr_gt.block(0,13,1,3).reshaped(3,1);

//         Eigen::Matrix3d R_c0_g  = R_b0_g * cam_model.R_c_b;
//         Eigen::Vector3d t_gc0_g = R_b0_g * cam_model.t_c_b + t_gb0_g;

//         // Next pose of the body and camera
//         Eigen::Matrix3d R_b1_g = next_gt.block(0,1,1,9).reshaped(3,3).transpose();
//         Eigen::Vector3d v_gb1_g = next_gt.block(0,10,1,3).reshaped(3,1);
//         Eigen::Vector3d t_gb1_g = next_gt.block(0,13,1,3).reshaped(3,1);

//         Eigen::Matrix3d R_c1_g  = R_b1_g * cam_model.R_c_b;
//         Eigen::Vector3d t_gc1_g = R_b1_g * cam_model.t_c_b + t_gb1_g;

//         /// Compute the iterative cam pose
//         Eigen::Matrix3d R_g_c1  = R_c1_g.transpose();
//         Eigen::Matrix3d R_c0_c1 = R_g_c1 * R_c0_g;
//         Eigen::Vector3d t_c0_c1 = R_g_c1 * (t_gc0_g - t_gc1_g);

//         // Load Depth Map
//         cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
//         cv::flip(depth_map, depth_map, 0);

//         // Vectorize Depth Map
//         depth_map.convertTo(depth_map, CV_64F); // Make sure that datatype is double
//         Eigen::RowVectorXd vectorized_depth_map = Eigen::Map<Eigen::RowVectorXd>(depth_map.ptr<double>(), depth_map.rows*depth_map.cols);

//         // // Get the estimate optical flow
//         // Eigen::MatrixXd vectorized_estimated_flow = cam_model.getEstimatedOF(vectorized_depth_map, R_c0_c1, t_c0_c1);
        
//         // Get the observed flow
//         cv::Mat observed_flow = DMU::load_flow(path_to_flow);   
//         Eigen::MatrixXd observed_of = cvFlow2Eigen(observed_flow);

//         // Compute w and b
//         Eigen::MatrixXd KR = cam_model.K * R_c0_c1;
//         Eigen::MatrixXd vectorized_w = KR * cam_model._vectorized_bearings;
//         Eigen::Vector3d b = cam_model.K * t_c0_c1;

//         // Create the vector for estimated depths
//         double _depths[cam_model._width*cam_model._width];
//         std::fill(_depths, _depths + cam_model._width * cam_model._width, 0.0);

//         // int pix_id = 1000;
//         auto start = std::chrono::high_resolution_clock::now();
//         for(int pix_id = 0; pix_id < cam_model._width*cam_model._height; ++pix_id){
//             Eigen::Vector2d p = cam_model._vectorized_pixels2.col(pix_id);
//             Eigen::Vector2d f = observed_of.col(pix_id);
//             Eigen::Vector3d w = vectorized_w.col(pix_id);

//             ceres::CostFunction* pixel_constraint = new ceres::NumericDiffCostFunction<SOLVER::PixelFlow, ceres::CENTRAL, 2, 1>(
//                 new SOLVER::PixelFlow(p, f, w, b));
            
//             // Set up the problem
//             ceres::Problem problem; 
//             problem.AddResidualBlock(pixel_constraint, nullptr, _depths+pix_id); // new ceres::CauchyLoss(1.0)

//             // Set up the solver options
//             ceres::Solver::Options options;
//             options.linear_solver_type = ceres::DENSE_QR;
//             options.minimizer_progress_to_stdout = false;
//             options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//             options.max_num_iterations = 50;
//             options.num_threads = 8;
//             options.min_trust_region_radius = 1e-12;

//             // Solve the problem
//             ceres::Solver::Summary summary;
//             ceres::Solve(options, &problem, &summary);
//         }

//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//         std::cout << "Execution time: " << duration << " microseconds" << std::endl;


//         break;
        
//         // std::cout << "GT : " << vectorized_depth_map(pix_id) << std::endl;
//         // std::cout << "ES : " << _depths[pix_id] << std::endl;
//         // std::cout << "Arbitrary : " << _depths[5000] << std::endl;

//         // ///////////////////// VISUALIZATIONS /////////////////////
//         // visualize_data(_data, idx, vectorized_estimated_flow);        
//     }
// }