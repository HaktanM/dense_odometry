#include <cuda_runtime.h>
#include "bootstrapper.hpp"


void DepthMapEstimator::print_data(){
    int size = 10 * sizeof(float);
    float *data = (float*)malloc(size);

    cudaMemcpy(data, d_pixels, size, cudaMemcpyDeviceToHost);
    std::cout << "Pixels : " << std::endl;
    for(int data_idx=0; data_idx<10; ++data_idx){
        std::cout << data[data_idx] << " ";
    }
    std::cout << std::endl;

    cudaMemcpy(data, d_bearings, size, cudaMemcpyDeviceToHost);
    std::cout << "Bearings : " << std::endl;
    for(int data_idx=0; data_idx<10; ++data_idx){
        std::cout << data[data_idx] << " ";
    }
    std::cout << std::endl;
}


__global__ void propapropagate_depth_prior_gpu(float *flow, float *depth_prior, float *depth_next, int height, int width){
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    long int totalPixCount = height*width;

    long int pix_idx = row_idx * width + col_idx;
    long int pixel_loc = 2 * pix_idx;

    // Get
    float flow_x = flow[pixel_loc];
    float flow_y = flow[pixel_loc+1];

    float propagated_x = col_idx + flow_x; // x is column index
    float propagated_y = row_idx + flow_y; // y is row index

    if((0<=propagated_x) && (propagated_x<=width) && (0<=propagated_y) && (propagated_y<=height)){
        int new_x = static_cast<int>(roundf(propagated_x));
        int new_y = static_cast<int>(roundf(propagated_y));

        int new_pix_idx = new_y * width + new_x;
        depth_next[new_pix_idx] = depth_prior[pix_idx]; // Copy paste the depth value
        depth_next[totalPixCount + new_pix_idx] = depth_prior[totalPixCount + pix_idx] * 2; // Copy past the sigma
    }
}

// void DepthMapEstimator::propagate_depth_prior(){
//     // Create a copy of the depth prior
//     cudaMemcpy(d_depth_prior_tmp, d_depth_prior, 2*_depth_size, cudaMemcpyDeviceToDevice);

//     // Initialize the depth map
//     cudaMemcpy(d_depth_prior, h_depth_prior, 2*_depth_size, cudaMemcpyHostToDevice);

//     // Define grid and block dimensions
//     dim3 threadsPerBlock(_width); // Each thread processes a single pixel
//     dim3 numBlocks(_height); // Each block processes an individual row

//     // Propagate the depth prior to be used in the next iteration
//     propapropagate_depth_prior_gpu<<<numBlocks, threadsPerBlock>>>(d_flow, d_depth_prior_tmp, d_depth_prior, _height, _width);
// }



__global__ void compute_depth_gpu(float* depth, float* depth_prior, float* depth_next, float* flow, float* pixels, float* bearings, float* KR, float* b, int height, int width) {
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    long int totalPixCount = height*width;

    long int pix_idx = row_idx * width + col_idx;
    long int pixel_loc = 2 * pix_idx;

    float flow_x = flow[pixel_loc];
    float flow_y = flow[pixel_loc+1];

    float px = pixels[pixel_loc];
    float py = pixels[pixel_loc+1];

    float a1 = flow_x + px;
    float a2 = flow_y + py;

    float bearing0 = bearings[pixel_loc];
    float bearing1 = bearings[pixel_loc + 1];

    float KR_00 = KR[0];
    float KR_10 = KR[1];
    float KR_20 = KR[2];

    float KR_01 = KR[3];
    float KR_11 = KR[4];
    float KR_21 = KR[5];

    float KR_02 = KR[6];
    float KR_12 = KR[7];
    float KR_22 = KR[8];

    float w1 = KR_00 * bearing0 + KR_01 * bearing1 + KR_02;
    float w2 = KR_10 * bearing0 + KR_11 * bearing1 + KR_12;
    float w3 = KR_20 * bearing0 + KR_21 * bearing1 + KR_22;

    float b1 = b[0];
    float b2 = b[1];
    float b3 = b[2];

    float z_num = (a1 * b3 - b1)*(w1 - a1 * w3) + (a2 * b3 - b2)*(w2 - a2 * w3);
    float z_deno = (w1 - a1 * w3)*(w1 - a1 * w3) + (w2 - a2 * w3)*(w2 - a2 * w3);

    float pd_new = z_num / z_deno;          // pd stands for pixel depth
    float pd_prior = depth_prior[pix_idx];  // pd stands for pixel depth

    float sigma_new = 0.5;
    float sigma_prior = depth_prior[totalPixCount + pix_idx];

    float sigma_sum = sigma_new + sigma_prior;
    float pd = (sigma_prior * pd_new + sigma_new * pd_prior) / sigma_sum;

    float sigma = sigma_new * sigma_prior / sigma_sum;

    depth[pix_idx] = pd;

    depth_prior[pix_idx] = pd;
    depth_prior[totalPixCount + pix_idx] = sigma;


    float propagated_x = col_idx + flow_x; // x is column index
    float propagated_y = row_idx + flow_y; // y is row index

    if((0<=propagated_x) && (propagated_x<=width) && (0<=propagated_y) && (propagated_y<=height)){
        int new_x = static_cast<int>(roundf(propagated_x));
        int new_y = static_cast<int>(roundf(propagated_y));

        int new_pix_idx = new_y * width + new_x;
        depth_next[new_pix_idx] = w3 * pd + b3; // Copy paste the depth value
        depth_next[totalPixCount + new_pix_idx] = depth_prior[totalPixCount + pix_idx] * w3 * w3 * 2; // Copy past the sigma
    }
}

void DepthMapEstimator::compute_depth_with_sigma(float* h_depth, float* h_depth_sigma, float* h_flow, float* h_KR, float* h_b){
    // Load parameters to device
    cudaMemcpy(d_flow, h_flow, _pixel_coord_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_KR, h_KR, _KR_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, _b_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(_width); // Each thread processes a single pixel
    dim3 numBlocks(_height); // Each block processes an individual row

    // Initialize the propagated depth map
    cudaMemcpy(d_depth_prior_next, h_depth_prior, 2*_depth_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_depth_prior, h_depth_prior, 2*_depth_size, cudaMemcpyHostToDevice);

    // Compute the estimated depth map on gpu
    compute_depth_gpu<<<numBlocks, threadsPerBlock>>>(d_depth, d_depth_prior, d_depth_prior_next, d_flow, d_pixels, d_bearings, d_KR, d_b, _height, _width);

    // Load depth to host
    cudaMemcpy(h_depth, d_depth, _depth_size, cudaMemcpyDeviceToHost);

    // Load depth uncertantiy to host
    cudaMemcpy(h_depth_sigma, d_depth_prior+_width*_height, _depth_size, cudaMemcpyDeviceToHost);

    // Propagate the depth map
    cudaMemcpy(d_depth_prior, d_depth_prior_next, 2*_depth_size, cudaMemcpyDeviceToDevice);

    // Sync the device to ensure kernel execution is complete
    cudaDeviceSynchronize();
}





__global__ void compute_flow_gpu(float* depth, float* pixels, float* bearings, float* KR, float* b, float* flow, int width, int height) {

    long int pix_idx = blockIdx.x * blockDim.x + threadIdx.x;
    long int pixel_loc = 2 * pix_idx;

    float d = depth[pix_idx];

    float px = pixels[pixel_loc];
    float py = pixels[pixel_loc+1];

    float bearing0 = bearings[pixel_loc];
    float bearing1 = bearings[pixel_loc + 1];

    float KR_00 = KR[0];
    float KR_10 = KR[1];
    float KR_20 = KR[2];

    float KR_01 = KR[3];
    float KR_11 = KR[4];
    float KR_21 = KR[5];

    float KR_02 = KR[6];
    float KR_12 = KR[7];
    float KR_22 = KR[8];

    float w1 = KR_00 * bearing0 + KR_01 * bearing1 + KR_02;
    float w2 = KR_10 * bearing0 + KR_11 * bearing1 + KR_12;
    float w3 = KR_20 * bearing0 + KR_21 * bearing1 + KR_22;

    float b1 = b[0];
    float b2 = b[1];
    float b3 = b[2];

    float flow_x = (w1*d + b1) / (w3*d + b3) - px;
    float flow_y = (w2*d + b2) / (w3*d + b3) - py;

    flow[pixel_loc]     = flow_x;
    flow[pixel_loc+1]   = flow_y;

    return;
}

void DepthMapEstimator::compute_optical_flow(float* depth, float* flow, float* KR, float* b){
    // Define grid and block dimensions
    dim3 threadsPerBlock(_width);
    dim3 numBlocks(_height);

    // Launch kernel to generate random values
    compute_flow_gpu<<<numBlocks, threadsPerBlock>>>(depth, d_pixels, d_bearings, KR, b, flow, _width, _height);

    // Sync the device to ensure kernel execution is complete
    cudaDeviceSynchronize();
}


