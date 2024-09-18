#include <cuda_runtime.h>
#include "gpu_manager.hpp"


void ManagerGPU::print_data(){
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



__global__ void COMPUTE_DEPTH_MAP_GPU(float* depth, float* depth_next, float* flow, float* pixels, float* bearings, float* KR, float* b, int height, int width) {
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

    // New computed depth and its covariance
    float pd_new = z_num / z_deno;          // pd stands for pixel depth
    float sigma_new = 0.5;

    // We have a depth prior from previous iterations
    float pd_prior = depth[pix_idx];        // pd stands for pixel depth
    float sigma_prior = depth[totalPixCount + pix_idx];

    float sigma_sum = sigma_new + sigma_prior;
    float pd = (sigma_prior * pd_new + sigma_new * pd_prior) / sigma_sum;

    float sigma = sigma_new * sigma_prior / sigma_sum;

    depth[pix_idx] = pd;
    depth[totalPixCount + pix_idx] = sigma;

    float propagated_x = col_idx + flow_x; // x is column index
    float propagated_y = row_idx + flow_y; // y is row index

    if((0<=propagated_x) && (propagated_x<=width) && (0<=propagated_y) && (propagated_y<=height)){
        int new_x = static_cast<int>(roundf(propagated_x));
        int new_y = static_cast<int>(roundf(propagated_y));

        int new_pix_idx = new_y * width + new_x;
        depth_next[new_pix_idx] = w3 * pd + b3; // Copy paste the depth value
        depth_next[totalPixCount + new_pix_idx] = depth[totalPixCount + pix_idx] * w3 * w3 * 2; // Copy past the sigma
    }
}

void ManagerGPU::refineDepthMap(float* h_depth, float* h_depth_sigma, float* h_flow, float* h_KR, float* h_b){
    // Load parameters to device
    cudaMemcpy(d_flow, h_flow, _pixel_coord_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_KR, h_KR, _KR_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, _b_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(_width); // Each thread processes a single pixel
    dim3 numBlocks(_height); // Each block processes an individual row

    // Initialize the propagated depth map
    cudaMemcpy(d_depth_next, h_depth_prior, 2*_depth_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_depth_prior, h_depth_prior, 2*_depth_size, cudaMemcpyHostToDevice);

    // Compute the estimated depth map on gpu
    COMPUTE_DEPTH_MAP_GPU<<<numBlocks, threadsPerBlock>>>(d_depth, d_depth_next, d_flow, d_pixels, d_bearings, d_KR, d_b, _height, _width);

    // Load depth to host
    cudaMemcpy(h_depth, d_depth, _depth_size, cudaMemcpyDeviceToHost);

    // Load depth uncertantiy to host
    cudaMemcpy(h_depth_sigma, d_depth+_width*_height, _depth_size, cudaMemcpyDeviceToHost);

    // Sync the device to ensure kernel execution is complete
    cudaDeviceSynchronize();
}

void ManagerGPU::propagateDepth(){
    // Propagate the depth map
    cudaMemcpy(d_depth, d_depth_next, 2*_depth_size, cudaMemcpyDeviceToDevice);

    // Sync the device to ensure kernel execution is complete
    cudaDeviceSynchronize();
}



__global__ void GET_FLOW_RES_GPU(float *flow_residual, float* depth, float* flow, float* pixels, float* bearings, float* KR, float* b, int height, int width) {
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    long int pix_idx = row_idx * width + col_idx;
    long int pixel_loc = 2 * pix_idx;

    float flow_x = flow[pixel_loc];
    float flow_y = flow[pixel_loc+1];

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

    float d = depth[pix_idx];

    float estimated_flow_x = (w1*d + b1) / (w3*d + b3) - px;
    float estimated_flow_y = (w2*d + b2) / (w3*d + b3) - py;

    flow_residual[pixel_loc]   = estimated_flow_x - flow_x;
    flow_residual[pixel_loc+1] = estimated_flow_y - flow_y;
}


void ManagerGPU::getFowResidual(float* h_flow_residual, float* h_KR, float* h_b) const{
    // Load parameters to device
    cudaMemcpy(d_KR, h_KR, _KR_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, _b_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(_width); // Each thread processes a single pixel
    dim3 numBlocks(_height); // Each block processes an individual row

    // Compute the estimated depth map on gpu
    GET_FLOW_RES_GPU<<<numBlocks, threadsPerBlock>>>(d_flow_residual, d_depth, d_flow, d_pixels, d_bearings, d_KR, d_b, _height, _width);

    // Load flow_residual to host
    cudaMemcpy(h_flow_residual, d_flow_residual, _pixel_coord_size, cudaMemcpyDeviceToHost);

    // Sync the device to ensure kernel execution is complete
    cudaDeviceSynchronize();
}


__global__ void GET_ESTIMATED_FLOW_GPU(float* depth, float* estimated_flow, float* pixels, float* bearings, float* KR, float* b, int height, int width) {
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    long int pix_idx = row_idx * width + col_idx;
    long int pixel_loc = 2 * pix_idx;

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

    float d = depth[pix_idx];

    float estimated_flow_x = (w1*d + b1) / (w3*d + b3) - px;
    float estimated_flow_y = (w2*d + b2) / (w3*d + b3) - py;

    estimated_flow[pixel_loc] = estimated_flow_x;
    estimated_flow[pixel_loc+1] = estimated_flow_y;
}


void ManagerGPU::getEstimatedFlow(float* h_estimated_flow, float* h_KR, float* h_b) const{
    // Load parameters to device
    cudaMemcpy(d_KR, h_KR, _KR_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, _b_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(_width); // Each thread processes a single pixel
    dim3 numBlocks(_height); // Each block processes an individual row


    // Compute the estimated depth map on gpu
    GET_ESTIMATED_FLOW_GPU<<<numBlocks, threadsPerBlock>>>(d_depth, d_estimated_flow, d_pixels, d_bearings, d_KR, d_b, _height, _width);

    // Load depth to host
    cudaMemcpy(h_estimated_flow, d_estimated_flow, _pixel_coord_size, cudaMemcpyDeviceToHost);

    // Sync the device to ensure kernel execution is complete
    cudaDeviceSynchronize();

}




