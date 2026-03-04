cmake_minimum_required(VERSION 3.18)

project(ImageProcessingTool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA backend" ON)

# -------------------------
# Dependencies
# -------------------------

find_package(OpenCV REQUIRED)

if(USE_CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        add_compile_definitions(USE_CUDA)
    else()
        message(WARNING "CUDA not found. Disabling CUDA backend.")
        set(USE_CUDA OFF)
    endif()
endif()

# -------------------------
# Source Collection (Modular)
# -------------------------

# Automatically re-scan when new files are added
file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
        src/*.cu
    )
endif()

# -------------------------
# Executable
# -------------------------

add_executable(server
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

# -------------------------
# Include Directories
# -------------------------

target_include_directories(server PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../Crow/include
)

# -------------------------
# Link Libraries
# -------------------------

target_link_libraries(server PRIVATE
    ${OpenCV_LIBS}
)

if(USE_CUDA)
    target_link_libraries(server PRIVATE CUDA::cudart)
    set_target_properties(server PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()