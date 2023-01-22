cmake_minimum_required(VERSION 3.0.0)
project(TP VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(${PROJECT_NAME}
src/main.cpp)

include_directories(include lib/eigen)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
