#include <cuda_runtime.h>
#include "gpu_manager.hpp"


// __device__ function to multiply a HEIGHTxHIDDEN and a HIDDENxWIDTH matrix using 1D arrays
__device__ void MatrixMaltiplication(const float *A, const float *B, float *C, int HEIGHT, int HIDDEN, int WIDTH) {
    // Iterate over rows of A
    for (int row_idx = 0; row_idx < HEIGHT; row_idx++) {
        // Iterate over columns of B 
        for (int col_idx = 0; col_idx < WIDTH; col_idx++) {
            C[row_idx * WIDTH + col_idx] = 0;  // Initialize the result element
            // Perform the dot product of the i-th row of A and the j-th column of B
            for (int k = 0; k < HIDDEN; k++) {
                C[row_idx * WIDTH + col_idx] += A[row_idx * HIDDEN + k] * B[k * WIDTH + col_idx];
            }
        }
    }
}


// __device__ function to generate a skew-symmetric matrix from a 3D vector
__device__ void getSkew(const float *vec, float *SkewForm) {
    // Row 1
    SkewForm[0] = 0.0f;      SkewForm[1] = -vec[2];  SkewForm[2] = vec[1];
    // Row 2
    SkewForm[3] = vec[2];    SkewForm[4] = 0.0f;      SkewForm[5] = -vec[0];
    // Row 3
    SkewForm[6] = -vec[1];   SkewForm[7] = vec[0];    SkewForm[8] = 0.0f;
}


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
    cudaMemcpy(d_KR, h_KR, _K_size, cudaMemcpyHostToDevice);
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


__global__ void GET_FLOW_RES_GPU(float *flow_residual, float *jacobeans, float* depth, float* flow, float* pixels, float* bearings, float* K, float* KR, float* b, int height, int width) {
    int row_idx = blockIdx.x;
    int col_idx = threadIdx.x;

    // Get the related parametrs in proper format
    long int pix_idx = row_idx * width + col_idx;
    long int pixel_loc = 2 * pix_idx;

    float flow_x = flow[pixel_loc];
    float flow_y = flow[pixel_loc+1];

    float px = pixels[pixel_loc];
    float py = pixels[pixel_loc+1];

    float bearing0 = bearings[pixel_loc];
    float bearing1 = bearings[pixel_loc + 1];

    // K and KR were Eigen Matrices. When Eigen is converted to array, the conversion is made columnwise
    // However, we assume that the conversion is made rowwise. Hence, here we revert the arrays
    float K_Ptr[9]  = {K[0],  K[3],  K[6],    K[1],  K[4],  K[7],    K[2],  K[5],  K[8]};
    float KR_Ptr[9] = {KR[0], KR[3], KR[6],   KR[1], KR[4], KR[7],   KR[2], KR[5], KR[8]};

    float bearingPtr[3] = {bearing0, bearing1, 1.0f};
    float wPtr[3];

    MatrixMaltiplication(KR_Ptr, bearingPtr, wPtr, 3, 3, 1);

    float w1 = wPtr[0];
    float w2 = wPtr[1];
    float w3 = wPtr[2];

    float b1 = b[0];
    float b2 = b[1];
    float b3 = b[2];

    float d = depth[pix_idx];

    // Compute frequently used variables
    float pp1_nh = w1 * d + b1;   // propagated pixel, non homogeneous coordinates (pp3_nh is not equal to 1)
    float pp2_nh = w2 * d + b2;   // propagated pixel, non homogeneous coordinates
    float pp3_nh = w3 * d + b3;   // propagated pixel, non homogeneous coordinates

    // Get residual
    float estimated_flow_x = pp1_nh / pp3_nh - px;
    float estimated_flow_y = pp2_nh / pp3_nh - py;

    flow_residual[pixel_loc]   = estimated_flow_x - flow_x;
    flow_residual[pixel_loc+1] = estimated_flow_y - flow_y;


    ////// Get Jacobean
    float df_dw[6];    // 2x3 Jacobean of flow with respect to w
    float df_db[6];    // 2x3 Jacobean of flow with respect b

    float dw_drot[9];  // 3x3 Jacobean

    float df_drot[6];  // 2x3 Jacobean of flow with respect to rotation
    float df_dtra[6];  // 2x3 Jacobean of flow with respect to translation cam

    // Frequently used parameters
    float pp3_nh_2 = pp3_nh * pp3_nh;

    //// df_db
    // First column
    df_db[0] = 1.0f / pp3_nh;
    df_db[1] = 0.0f;
    df_db[2] = - pp1_nh / pp3_nh_2;
    // Second column
    df_db[3] = 0.0f;
    df_db[4] = 1.0f / pp3_nh;
    df_db[5] = - pp2_nh / pp3_nh_2;

    //// df_dw
    for(int i=0; i<6; i++){
        df_dw[i] = d * df_db[i];
    }
    
    //// dw_drot
    float bearingSkew[9];
    getSkew(bearingPtr, bearingSkew);
    MatrixMaltiplication(KR_Ptr, bearingSkew, dw_drot, 3, 3, 3);
    for(int i=0; i<9; i++){
        dw_drot[i] = - dw_drot[i];
    }

    //// df_drot  
    MatrixMaltiplication(df_dw, dw_drot, df_drot, 2, 3, 3);

    //// df_dtra
    MatrixMaltiplication(df_db, K_Ptr, df_dtra, 2, 3, 3);

    long int jacobean_idx = pix_idx * 6 * 2;

    // Load the derivative of fx with respect to state
    for(int i=0; i<3; i++){
        jacobeans[jacobean_idx + i] = df_drot[i];
    }   

    for(int i=0; i<3; i++){
        jacobeans[jacobean_idx + 3 + i] = df_dtra[i];
    }  

    // Load the derivative of fy with respect to state
    for(int i=0; i<3; i++){
        jacobeans[jacobean_idx + 6 + i] = df_drot[i + 3];
    }   

    for(int i=0; i<3; i++){
        jacobeans[jacobean_idx + 9 + i] = df_dtra[i + 3];
    }   
}


void ManagerGPU::getFowResidual(float* h_flow_residual, float *h_jacobean, float* h_K, float* h_KR, float* h_b) const{
    // Load parameters to device
    cudaMemcpy(d_K, h_K, _K_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_KR, h_KR, _K_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, _b_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(_width); // Each thread processes a single pixel
    dim3 numBlocks(_height); // Each block processes an individual row

    // Compute the estimated depth map on gpu
    GET_FLOW_RES_GPU<<<numBlocks, threadsPerBlock>>>(d_flow_residual, d_jacobean, d_depth, d_flow, d_pixels, d_bearings, d_K, d_KR, d_b, _height, _width);

    // Load flow_residual to host
    cudaMemcpy(h_flow_residual, d_flow_residual, _pixel_coord_size, cudaMemcpyDeviceToHost);

    // Load jacobean to host
    cudaMemcpy(h_jacobean, d_jacobean, _pixel_coord_size * 6, cudaMemcpyDeviceToHost);

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
    cudaMemcpy(d_KR, h_KR, _K_size, cudaMemcpyHostToDevice);
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



__global__ void TEST_MATRIX_MUL(float* A, float* B, float *C, int W, int Hidden, int H){
    MatrixMaltiplication(A,B,C,W,Hidden,H);
}


void ManagerGPU::testMatrixMultiplication(float* h_A, float* h_B, float *h_C, int W, int Hidden, int H){
    // Get memory from device to load input matrices
    float *d_A;
    float *d_B;
    float *d_C;

    int sizeA = W * Hidden * sizeof(float);
    int sizeB = Hidden * H * sizeof(float);
    int sizeC = W * H * sizeof(float);

    cudaMalloc((void**)&d_A, sizeA);  // Memory in Device (GPU)
    cudaMalloc((void**)&d_B, sizeB);  // Memory in Device (GPU)
    cudaMalloc((void**)&d_C, sizeC);  // Memory in Device (GPU)

    // Now load the matrices from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(1); // Each thread processes a single pixel
    dim3 numBlocks(1); // Each block processes an individual row

    // Compute the estimated depth map on gpu
    TEST_MATRIX_MUL<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, W, Hidden, H);

    // Load the result to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}