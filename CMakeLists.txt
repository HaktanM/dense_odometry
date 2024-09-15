cmake_minimum_required(VERSION 3.10)

# This project aims to find eigenvalues of a given matrix
project(DenseODOM C CUDA) # C CUDA is reqired to compile .cu files. Otherwise they are not compiled.

project(${PROJECT_NAME} VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the Eigen Library
find_package (Eigen3 REQUIRED)

# Find the OpenCV Library
find_package (OpenCV REQUIRED)

# Find Ceres Library for Optimization
find_package(Ceres REQUIRED)

# Enable CUDA support
find_package(CUDA REQUIRED)

# Add the include directory to the include path
include_directories(
    include
    gpu_kernels
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

# Find all .cpp files in the src directory
file(GLOB SOURCES "src/*.cpp" "gpu_kernels/*.cu")

# Create executable
add_executable(dene ${SOURCES})

# set(GPU_KERNELS "/home/hakito/cpp_scripts/dense_odometry/libgpu-kernels.so")

# Link the related libraries to our executable
target_link_libraries (dene 
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${GPU_KERNELS}
)

# set_target_properties(dene PROPERTIES CUDA_SEPARABLE_COMPILATION ON)