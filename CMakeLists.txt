cmake_minimum_required(VERSION 3.16)
project(torchlet CXX)

set(CMAKE_CXX_STANDARD 17) # try c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # force c++17
set(CMAKE_CXX_EXTENSIONS OFF) # force using -std=c++17

find_library(ACCELERATE Accelerate REQUIRED)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_library(torchlet 
src/tensor.cpp 
src/init.cpp
src/kernel.cpp 
src/linear.cpp
src/functional.cpp
src/iterator.cpp)


target_include_directories(torchlet 
PUBLIC ${PROJECT_SOURCE_DIR}/include
PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(torchlet PRIVATE ${ACCELERATE})

# Test
enable_testing()
add_subdirectory(tests)

#Benchmark
add_subdirectory(benchmarks)

# Main
add_executable(main main.cpp) # create an excecutable from main.cpp
target_link_libraries(main PRIVATE torchlet)

foreach(tgt IN ITEMS torchlet main)
    target_compile_options(${tgt} PRIVATE
      -O3
      -ffast-math 
      -fno-math-errno
      -mcpu=native  
      -Wall -Wextra -Wshadow -Wconversion
    )
  endforeach()