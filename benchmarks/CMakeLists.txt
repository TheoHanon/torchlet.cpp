add_executable(torchlet_bench 
        bench_mvb.cpp)

target_link_libraries(torchlet_bench PRIVATE torchlet ${ACCELERATE})
