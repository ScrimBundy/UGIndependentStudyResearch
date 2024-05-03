cmake_minimum_required(VERSION 3.5)
project(ISResearch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O2 -static")
set(CMAKE_CXX_FLAGS_DEBUG  "-g -O0 -static")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--whole-archive -lrt -lpthread -Wl,--no-whole-archive")

add_executable(ISResearch main.cpp)

target_link_libraries(ISResearch PRIVATE pthread rt)
