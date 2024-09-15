#pragma ONCE


#include <cuda_runtime.h>
#include <curand_kernel.h> // for random numbers
#include <iostream>

struct DepthMapEstimator{
public:
    DepthMapEstimator(float *h_pixels, float *h_bearings, int height, int width): _height(height), _width(width)
    {   

        _depth_size = _width * _height * sizeof(float);
        _pixel_coord_size = 2 * _width * _height * sizeof(float);
        _KR_size = 9 * sizeof(float);
        _b_size  = 3 * sizeof(float);


        cudaMalloc((void**)&d_pixels, _pixel_coord_size);  // Memory in Device (GPU)
        cudaMemcpy(d_pixels, h_pixels, _pixel_coord_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_bearings, _pixel_coord_size);  // Memory in Device (GPU)
        cudaMemcpy(d_bearings, h_bearings, _pixel_coord_size, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&d_depth, 2*_depth_size); // in d_depth, we both have the depth and its covariance
        cudaMalloc((void**)&d_depth_prior, 2*_depth_size); // in d_depth, we both have the depth and its covariance
        cudaMalloc((void**)&d_depth_prior_next, 2*_depth_size); // in d_depth, we both have the depth and its covariance

        cudaMalloc((void**)&d_flow, _pixel_coord_size); // Allocate memory

        cudaMalloc((void**)&d_KR, _KR_size);
        cudaMalloc((void**)&d_b, _b_size);

        // Initialize the depth prior
        h_depth_prior = (float*)malloc(sizeof(float) * 2 * _depth_size);;
        std::fill(h_depth_prior, h_depth_prior + _width*_height, 10.0f);  // 10.0 is just a dummy number
        std::fill(h_depth_prior + _width*_height, h_depth_prior + 2*_width*_height, 10000.0f);  // initial covariance of the depth prior is high
        // Now initialize the depth prior in device
        cudaMemcpy(d_depth_prior, h_depth_prior, 2*_depth_size, cudaMemcpyHostToDevice);
    }

    void print_data();
    void compute_depth_with_sigma(float* h_depth,float* h_depth_sigma, float* h_flow, float* h_KR, float* h_b);
    void compute_optical_flow(float* depth, float* flow, float* KR, float* b);

private:
    float *d_bearings;
    float *d_pixels;

    float *d_depth;
    float *d_depth_prior;
    float *d_depth_prior_next;
    float *d_flow;
    
    float *h_depth_prior;

    float *d_KR;  
    float *d_b;  


    int _height, _width;
    int _depth_size;
    int _pixel_coord_size;
    int _KR_size;
    int _b_size;

};
