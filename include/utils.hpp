#pragma once

#include <filesystem>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>  // For std::sort
#include <opencv2/opencv.hpp>

#include <fstream>

#include <Eigen/Dense>

#include <cmath> // Required to compute sin and cos


namespace DMU{ // DMU refers for Data Management Utils
    // A Utility to Load CSV Files
    void loadCSV(std::string path, std::map<long int, Eigen::MatrixXd> &content, const int cols);

    // A Utility to Loaf .flo Files
    cv::Mat load_flow(const std::string& path);

    // imgList takes the path to the images
    // Creates a list of the images in the folder, does NOT reads/loads images
    // Sorts the image names to feed them in proper order !!! VERY IMPORTANT
    // Creates the path to the image when asked. 
    class itemList{
    public:
        itemList(std::string path_to_data);     
        std::string getItemPath(int img_idx);    // given the index of the item, returns the global path to the item
        std::string getItemName(int img_idx);    // given the index of the item, returns name of the item
        std::string getItemPathFromName(std::string item_name);
         
        int itemCount(){
            return item_ids.size();
        }
    private:
        std::string main_path;                  // the global path of the image folder  
        std::string file_extension;             // determined automatically, .png, .jpg, .tiff etc
        bool has_zero_padding = true;           // determined automatically, true -> 001.png, 002.png   false-> 1.png, 2.png, 3.png
        int NUM_ZEROS = 0;                      // determined automatically, if has zero padding, what is the length of the name
        std::vector<long int> item_ids;

    }; // class itemList

    class dataHandler{
    public:
        dataHandler(std::string main_path);

        std::string flow_folder_name  = "RaftOF";
        std::string depth_folder_name = "Depth";
        std::string image_folder_name = "Images";

        std::string flow_folder;
        std::string depth_folder;
        std::string image_folder;

        // itemList flow_list;
        std::string gt_file_name   =  "synched_gt.csv";
        std::string imu_file_name  =  "preintegrated_imu.csv";

        std::string path_to_gt;
        std::string path_to_imu;

        std::map<long int, Eigen::MatrixXd> imu_data, gt_data;

        itemList flowList, depthList, imgList;
    }; // class dataHandler


    
} // namespace DMU

 
namespace VU{ // VU Stands For Visualization Tools
    void makeColorWheel(std::vector<cv::Vec3b> &colorwheel);
    cv::Mat flowToColor(const cv::Mat &flow);
    cv::Mat warpFlow(const cv::Mat& img, const cv::Mat& flow);
} // namespace VU


namespace LU{ // LU stands for Lie Algebra Utils
    static double _tolerance{1e-10};
    Eigen::Matrix3d Skew(Eigen::Vector3d vec);
    Eigen::Matrix3d exp_SO3(Eigen::Vector3d psi);
} // namespace LU