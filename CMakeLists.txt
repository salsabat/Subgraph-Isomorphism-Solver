cmake_minimum_required(VERSION 3.18)
project(SubgraphSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common sources
set(COMMON_SOURCES
    src/main.cpp
    src/solvers/cpu_solver.cpp
    src/solvers/vf2_solver.cpp
)

# ML solver sources
set(ML_SOURCES
    src/solvers/ml_solver.cpp
)

# Try to find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/include/onnxruntime
          /usr/include/onnxruntime
          /opt/homebrew/include/onnxruntime
          $ENV{ONNXRUNTIME_ROOT}/include)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS /usr/local/lib
          /usr/lib
          /opt/homebrew/lib
          $ENV{ONNXRUNTIME_ROOT}/lib)

# Try to enable CUDA with modern approach
enable_language(CUDA)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, building with GPU support")
    set(CUDA_SOURCES src/solvers/optdfs_solver.cu)
else()
    message(STATUS "CUDA not found, building CPU-only version")
    set(CUDA_SOURCES "")
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "ONNX Runtime found, building with ML support")
    set(ENABLE_ML ON)
else()
    message(STATUS "ONNX Runtime not found, building without ML support")
    set(ENABLE_ML OFF)
    set(ML_SOURCES "")
endif()

# Create executable
add_executable(solver ${COMMON_SOURCES} ${CUDA_SOURCES} ${ML_SOURCES})

# Set CUDA properties if available
if(CUDAToolkit_FOUND)
    set_target_properties(solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
    )
    target_link_libraries(solver CUDA::cudart)
endif()

# Set ML properties if available
if(ENABLE_ML)
    target_include_directories(solver PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(solver ${ONNXRUNTIME_LIBRARY})
endif()

target_include_directories(solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Print build configuration
message(STATUS "Build configuration:")
message(STATUS "  CUDA support: ${CUDAToolkit_FOUND}")
message(STATUS "  ML support: ${ENABLE_ML}")
if(ENABLE_ML)
    message(STATUS "  ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()
