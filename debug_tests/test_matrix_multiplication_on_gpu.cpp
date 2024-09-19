#include <iostream>
#include <chrono>
#include "manager.hpp"
int main(){

    std::string path_to_data = "/home/hakito/python_scripts/AirSim/Data3";
    
    std::shared_ptr<Manager> manager = std::make_shared<Manager>(path_to_data);

    // First check if matrix multiplication function is working fine
    // Matrix dimensions
    int W = 3;       // Rows of A
    int Hidden = 1;  // Columns of A and Rows of B
    int H = 3;       // Columns of B

    // Create random matrices using OpenCV
    cv::Mat matA = cv::Mat(W, Hidden, CV_32F);
    cv::Mat matB = cv::Mat(Hidden, H, CV_32F);
    cv::Mat matC = cv::Mat(W, H, CV_32F); // Result matrix

    cv::theRNG().state = static_cast<uint64_t>(cv::getTickCount());
    // Fill matrices with random values
    cv::randu(matA, cv::Scalar(0), cv::Scalar(10)); // Random values between 0 and 10
    cv::randu(matB, cv::Scalar(0), cv::Scalar(10)); // Random values between 0 and 10

    // Convert matrices to float arrays
    float* h_A = (float*)matA.data;
    float* h_B = (float*)matB.data;
    float* h_C = (float*)matC.data;

    manager->_GPUhandler.testMatrixMultiplication(h_A, h_B, h_C, W, Hidden, H);
    
    cv::Mat C_GPU(W, H, CV_32F, h_C);
    cv::Mat C_CPU = matA * matB;

    // Print the result matrix
    std::cout << "C_GPU:\n" << C_GPU << std::endl;
    std::cout << "C_CPU:\n" << C_CPU << std::endl;

    

}