cmake_minimum_required(VERSION 3.10)

project(openMP C)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wpedantic -fopenmp -lm")

set(SRC_DIR src)
set(LIB_DIR include)

set(HEADERS
        ${LIB_DIR}/hpc.h
        ${LIB_DIR}/network.h
        ${LIB_DIR}/utilities.h
)

set(SOURCES
        ${SRC_DIR}/main.c
        ${SRC_DIR}/network.c
        ${SRC_DIR}/utilities.c
)



add_executable(openMP ${SOURCES} ${HEADERS})

target_link_libraries(openMP m)

target_include_directories(openMP PRIVATE ${LIB_DIR})
