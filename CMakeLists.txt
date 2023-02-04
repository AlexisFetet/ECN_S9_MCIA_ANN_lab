cmake_minimum_required(VERSION 3.0.0)
project(TP VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(TRAINING_IMAGES="${CMAKE_CURRENT_LIST_DIR}/data/train-images-idx3-ubyte")
add_compile_definitions(TRAINING_LABELS="${CMAKE_CURRENT_LIST_DIR}/data/train-labels-idx1-ubyte")
add_compile_definitions(TEST_IMAGES="${CMAKE_CURRENT_LIST_DIR}/data/t10k-images-idx3-ubyte")
add_compile_definitions(TEST_LABELS="${CMAKE_CURRENT_LIST_DIR}/data/t10k-labels-idx1-ubyte")



include(CTest)
enable_testing()

add_executable(${PROJECT_NAME}
src/main.cpp
src/layer.cpp
src/network.cpp
src/parser.c)

find_package(OpenMP REQUIRED)

include_directories(lib/eigen)
include_directories(include)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
