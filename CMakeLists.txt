cmake_minimum_required(VERSION 3.0.0)
project(TP VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

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
