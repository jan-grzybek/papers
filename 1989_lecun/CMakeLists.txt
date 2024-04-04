
cmake_minimum_required(VERSION 3.27)
project(lecun C)

set(CMAKE_C_STANDARD 23)

add_executable(lecun main.c)
target_link_libraries(lecun m)
