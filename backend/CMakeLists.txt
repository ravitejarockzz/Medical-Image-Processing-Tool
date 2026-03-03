cmake_minimum_required(VERSION 3.18)

project(MedicalImageApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

# Try to find CUDA (optional)
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found. GPU acceleration enabled.")
    set(USE_CUDA ON)
else()
    message(STATUS "CUDA not found. Building CPU-only version.")
    set(USE_CUDA OFF)
endif()

# Base executable (always compiled)
add_executable(server
    src/main.cpp
    src/CPUProcessor.cpp
)

# If CUDA is available, add CUDA sources
if (USE_CUDA)
    target_sources(server PRIVATE src/CUDAProcessor.cu)

    set_target_properties(server PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 89   # RTX 4050
    )

    target_link_libraries(server PRIVATE CUDA::cudart)

    target_compile_definitions(server PRIVATE USE_CUDA)
endif()

target_include_directories(server PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../Crow/include
)

target_link_libraries(server PRIVATE
    ${OpenCV_LIBS}
)