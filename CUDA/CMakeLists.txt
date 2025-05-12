cmake_minimum_required(VERSION 3.30)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(SRC_DIR src)
set(LIB_DIR include)

set(HEADERS
        ${LIB_DIR}/hpc.h
)

set(SOURCES
        ${SRC_DIR}/main.cu
)

add_executable(CUDA ${SOURCES} ${HEADERS})

include_directories(${LIB_DIR})


target_link_libraries(CUDA m)

set_target_properties(CUDA PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
