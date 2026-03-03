cmake_minimum_required(VERSION 3.18)

project(MedicalImageApp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(server
    src/main.cpp
    src/CPUProcessor.cpp
    src/CUDAProcessor.cu
)

set_target_properties(server PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(server PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../Crow/include
)

target_link_libraries(server PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
)