cmake_minimum_required(VERSION 3.10)

project(APAI_ff_nn C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wpedantic -fopenmp -lm")

set(SRC_DIR src)
set(LIB_DIR include)

set(SOURCES
    ${SRC_DIR}/main.c
    ${SRC_DIR}/utilities.c
)

set(HEADERS
    ${LIB_DIR}/hpc.h
    ${LIB_DIR}/utilities.h
)


add_executable(APAI_ff_nn ${SOURCES} ${HEADERS})

target_link_libraries(APAI_ff_nn m)

target_include_directories(APAI_ff_nn PRIVATE ${LIB_DIR})