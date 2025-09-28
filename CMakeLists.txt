cmake_minimum_required(VERSION 3.16)
project(torchlet CXX)

set(CMAKE_CXX_STANDARD 17) # try c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # force c++17
set(CMAKE_CXX_EXTENSIONS OFF) # force using -std=c++17

add_executable(main main.cpp) # create an excecutable from main.cpp


add_library(torchlet 
    src/tensor.cpp 
    src/init.cpp
    src/kernel.cpp 
    src/linear.cpp)

target_include_directories(torchlet 
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(main PRIVATE torchlet)