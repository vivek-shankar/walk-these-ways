cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)

include_directories(include)
include_directories(include/lcm_types)

link_directories(lib)

add_compile_options(-std=c++11)

set(EXTRA_LIBS -pthread libunitree_legged_sdk_arm64.so lcm)

set(CMAKE_CXX_FLAGS "-O3")

# one pc one process
add_executable(lcm_position examples/lcm_position.cpp)
target_link_libraries(lcm_position ${EXTRA_LIBS})




