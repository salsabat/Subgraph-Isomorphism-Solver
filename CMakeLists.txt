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

# Try to enable CUDA with modern approach
enable_language(CUDA)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found, building with GPU support")

    set(CUDA_SOURCES
        src/solvers/optdfs_solver.cu
    )

    add_executable(solver ${COMMON_SOURCES} ${CUDA_SOURCES})

    set_target_properties(solver PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
    )

    target_link_libraries(solver CUDA::cudart)
else()
    message(STATUS "CUDA not found, building CPU-only version")
    add_executable(solver ${COMMON_SOURCES})
endif()

target_include_directories(solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
