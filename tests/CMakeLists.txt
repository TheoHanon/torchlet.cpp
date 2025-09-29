add_executable(torchlet_tests 
    kernel_test.cpp)

target_link_libraries(
    torchlet_tests PRIVATE gtest_main torchlet
)

include(GoogleTest)
gtest_discover_tests(torchlet_tests)

