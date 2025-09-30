add_executable(torchlet_tests 
    kernel_test.cpp
    tensor_test.cpp
    linear_test.cpp
    init_test.cpp)

target_link_libraries(
    torchlet_tests PRIVATE gtest_main torchlet
)

target_include_directories(torchlet_tests 
    PRIVATE ${PROJECT_CURRENT_SOURCE_DIR}/tests/utils
)

include(GoogleTest)
gtest_discover_tests(torchlet_tests)

