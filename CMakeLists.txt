cmake_minimum_required(VERSION 3.0.0)
project(TP VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(TRAINING_IMAGES="${CMAKE_CURRENT_LIST_DIR}/data/training/train-images-idx3-ubyte")
add_compile_definitions(TRAINING_LABELS="${CMAKE_CURRENT_LIST_DIR}/data/training/train-labels-idx1-ubyte")
add_compile_definitions(TEST_IMAGES="${CMAKE_CURRENT_LIST_DIR}/data/test/t10k-images-idx3-ubyte")
add_compile_definitions(TEST_LABELS="${CMAKE_CURRENT_LIST_DIR}/data/test/t10k-labels-idx1-ubyte")

include(CTest)
enable_testing()

add_executable(${PROJECT_NAME}
src/main.cpp
src/images.cpp
src/labels.cpp
src/parser.c)

include_directories(include)
include_directories(lib/eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
