#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>

// CUDA libraries
#include <cuda_runtime.h>
#include "bootstrapper.hpp"  // Include the header file

#include "utils.hpp"
#include "cam_model.hpp"

#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;


std::vector<cv::Mat> error_histograms(cv::Mat depth_map, cv::Mat estimated_depth_map, long int totalPixCount){

    // Copmute depth estimation error
    cv::Mat depth_map_err = estimated_depth_map - depth_map;

    // Compute percentage error
    cv::Mat depth_map_err_percentage;
    cv::divide(depth_map_err, depth_map, depth_map_err_percentage);
    depth_map_err_percentage = depth_map_err_percentage * 100;

    // Define the number of bins
    int histSize = 200; // Number of intensity levels for grayscale image

    // Set the ranges for pixel values (0-255 for grayscale)
    float range[] = { -10, 10 };
    const float* histRange = { range };

    // Create a cv::Mat to store the histogram
    cv::Mat hist_err, hist_err_perc;

    // Calculate the histogram
    cv::calcHist(&depth_map_err, 1, 0, cv::Mat(), hist_err, 1, &histSize, &histRange);
    cv::calcHist(&depth_map_err_percentage, 1, 0, cv::Mat(), hist_err_perc, 1, &histSize, &histRange);

    // Convert the histogram data to a std::vector for matplotlibcpp
    std::vector<float> histData_err(histSize);
    for (int i = 0; i < histSize; i++) {
        histData_err[i] = hist_err.at<float>(i) / ((float) totalPixCount);
    }

    // Convert the histogram data to a std::vector for matplotlibcpp
    std::vector<float> histData_err_perc(histSize);
    for (int i = 0; i < histSize; i++) {
        histData_err_perc[i] = hist_err_perc.at<float>(i) / ((float) totalPixCount);
    }

    // Prepare x-axis values for the histogram
    std::vector<float> xValues(histSize);
    for (int i = 0; i < histSize; i++) {
        xValues[i] = (i * (range[1] - range[0]) / histSize) + range[0];
    }

    // Plot the histogram using matplotlibcpp
    plt::figure_size(640, 512); // Set figure size
    plt::plot(xValues,histData_err);
    plt::title("Depth Error Histogram");
    plt::xlabel("Error");
    plt::ylabel("PDF");
    plt::ylim(0.0, 0.5);
    plt::grid(true); // Add grid for better visibility
    
    // Render the plot to a buffer
    plt::save("DepthErrorHist.png"); // Save to a temporary file
    cv::Mat depth_err_hist_img = cv::imread("DepthErrorHist.png"); // Load the saved image

    // cv::imshow("Depth Error Histogram", histogramImage);

    plt::clf();
    plt::figure_size(640, 512); // Set figure size
    plt::plot(xValues,histData_err_perc);
    plt::title("Depth Error Percentage Histogram");
    plt::xlabel("Error");
    plt::ylabel("PDF");
    plt::ylim(0.0, 0.2);
    plt::grid(true); // Add grid for better visibility
    
    // Render the plot to a buffer
    plt::save("DepthErrorPercentageHist.png"); // Save to a temporary file
    cv::Mat depth_err_perc_hist_img = cv::imread("DepthErrorPercentageHist.png"); // Load the saved image

    std::vector<cv::Mat> histograms;
    histograms.push_back(depth_err_hist_img);
    histograms.push_back(depth_err_perc_hist_img);

    return histograms;
}
int main() {
    CamModel cam_model;

    std::string main_path = "/home/hakito/python_scripts/AirSim/Data3";
    DMU::dataHandler _data(main_path);

    int width = cam_model._width;
    int height = cam_model._height;
    long int totalPixCount = width*height;

    // Size of the required memory
    int depth_size = width * height * sizeof(float);
    int flow_size = 2 * width * height * sizeof(float);
    int pixel_coord_size = 2 * width * height * sizeof(float);
    
    // Allocate memory for depth
    float *depthPtr = (float*)malloc(depth_size);      // Memory in Host
    float *depthSigmaPtr = (float*)malloc(depth_size);      // Memory in Host

    // Get pixel coordinates, and load to device.
    Eigen::MatrixXf pixels = cam_model._vectorized_pixels2.cast<float>();  // Get bearings from camera model
    float* pixelsPtr = pixels.data(); // Convert bearing to float array

    // Get bearings (K_inv * p), and load to device.
    Eigen::MatrixXf bearings = cam_model._vectorized_bearings.topRows(2).cast<float>();  // Get bearings from camera model
    float* bearingPtr = bearings.data(); // Convert bearing to float array

    
    DepthMapEstimator depthMapEstimator(pixelsPtr, bearingPtr, height, width);
    

    for(int idx = 300; idx<_data.flowList.itemCount()-2; ++idx){
        auto start = std::chrono::high_resolution_clock::now();

        // Get the related data    
        std::string path_to_flow, path_to_img_curr, path_to_img_next, path_to_depth, curr_timestamp_string, next_timestamp_string;
        path_to_flow                =   _data.flowList.getItemPath(idx);
        curr_timestamp_string       =   _data.flowList.getItemName(idx);
        next_timestamp_string       =   _data.flowList.getItemName(idx+1);
        path_to_img_curr            =   _data.imgList.getItemPathFromName(curr_timestamp_string);
        path_to_img_next            =   _data.imgList.getItemPathFromName(next_timestamp_string);
        path_to_depth               =   _data.depthList.getItemPathFromName(curr_timestamp_string);

        long int curr_timestamp = std::stol(curr_timestamp_string);
        long int next_timestamp = std::stol(next_timestamp_string);

        Eigen::MatrixXd curr_gt = _data.gt_data.at(curr_timestamp);
        Eigen::MatrixXd next_gt = _data.gt_data.at(next_timestamp);
        
        /// Get the pose of the camera with respect to global navigation frame
        // Current pose of the body and camera
        Eigen::Matrix3d R_b0_g = curr_gt.block(0,1,1,9).reshaped(3,3).transpose();
        Eigen::Vector3d v_gb0_g = curr_gt.block(0,10,1,3).reshaped(3,1);
        Eigen::Vector3d t_gb0_g = curr_gt.block(0,13,1,3).reshaped(3,1);

        Eigen::Matrix3d R_c0_g  = R_b0_g * cam_model.R_c_b;
        Eigen::Vector3d t_gc0_g = R_b0_g * cam_model.t_c_b + t_gb0_g;

        // Next pose of the body and camera
        Eigen::Matrix3d R_b1_g = next_gt.block(0,1,1,9).reshaped(3,3).transpose();
        Eigen::Vector3d v_gb1_g = next_gt.block(0,10,1,3).reshaped(3,1);
        Eigen::Vector3d t_gb1_g = next_gt.block(0,13,1,3).reshaped(3,1);

        Eigen::Matrix3d R_c1_g  = R_b1_g * cam_model.R_c_b;
        Eigen::Vector3d t_gc1_g = R_b1_g * cam_model.t_c_b + t_gb1_g;

        /// Compute the iterative cam pose
        Eigen::Matrix3d R_g_c1  = R_c1_g.transpose();
        Eigen::Matrix3d R_c0_c1 = R_g_c1 * R_c0_g;
        Eigen::Vector3d t_c0_c1 = R_g_c1 * (t_gc0_g - t_gc1_g);

        Eigen::Matrix3f KR = (cam_model.K * R_c0_c1).cast<float>();
        Eigen::Vector3f b  = (cam_model.K * t_c0_c1).cast<float>();

        float* KR_Ptr = KR.data();
        float* bPtr = b.data();

        /// Get the observed flow
        cv::Mat observed_flow = DMU::load_flow(path_to_flow);
        float* flowPtr = observed_flow.ptr<float>();

        /////////////////// GET THE DEPTH MAP ///////////////////
        depthMapEstimator.compute_depth_with_sigma(depthPtr, depthSigmaPtr, flowPtr, KR_Ptr, bPtr);
        /////////////////////////////////////////////////////////

        // End time measurement
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration in milliseconds
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Output the elapsed time in milliseconds
        std::cout << "Time taken: " << elapsed.count() << " milliseconds" << std::endl;

        

        


        //// VISUALIZE
        // Depth Map       
        cv::Mat depth_map = cv::imread(path_to_depth, cv::IMREAD_UNCHANGED);
        cv::flip(depth_map, depth_map, 0);
        cv::Mat depth_map_vis = depth_map / 100.0 * 255.0;
        depth_map_vis.convertTo(depth_map_vis, CV_8U);
        cv::cvtColor(depth_map_vis, depth_map_vis, cv::COLOR_GRAY2BGR);
        cv::putText(depth_map_vis, "Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 255, 100), 2);

        // Estimated Depth Map
        cv::Mat estimated_depth(height, width, CV_32F, depthPtr);
        cv::Mat estimated_depth_vis = estimated_depth / 100.0 * 255.0;
        estimated_depth_vis.convertTo(estimated_depth_vis, CV_8U);
        cv::cvtColor(estimated_depth_vis, estimated_depth_vis, cv::COLOR_GRAY2BGR);
        cv::putText(estimated_depth_vis, "Estimated Depth Map", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 255, 100), 2);

        // Covariance of the Estimated depth
        cv::Mat depth_sigma(height, width, CV_32F, depthSigmaPtr);
        cv::Mat depth_sigma_vis = depth_sigma * 2 * 220;
        depth_sigma_vis.convertTo(depth_sigma_vis, CV_8U);
        cv::cvtColor(depth_sigma_vis, depth_sigma_vis, cv::COLOR_GRAY2BGR);
        cv::putText(depth_sigma_vis, "Depth Variance", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100, 255, 100), 2);

        // Image Frame
        cv::Mat frame = cv::imread(path_to_img_curr);

        // Depth Error
        std::vector<cv::Mat> histograms = error_histograms(depth_map, estimated_depth, totalPixCount);
        cv::Mat depth_err_hist_img = histograms.at(0);
        cv::Mat depth_err_perc_hist_img = histograms.at(1);

        cv::Mat out_up, out_down, out;

        cv::hconcat(frame, depth_sigma_vis, out_up);
        cv::hconcat(out_up, depth_err_hist_img, out_up);

        cv::hconcat(depth_map_vis, estimated_depth_vis, out_down);
        cv::hconcat(out_down, depth_err_perc_hist_img, out_down);

        cv::vconcat(out_up, out_down, out);

        cv::imshow("out", out);

        std::string filename = "/home/hakito/cpp_scripts/dense_odometry/out_img3/" + std::to_string(idx) + ".png";
        cv::imwrite(filename, out);

        cv::waitKey(10);
    }

    return 0;
}